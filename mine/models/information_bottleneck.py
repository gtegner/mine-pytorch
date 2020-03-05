import itertools
import torch
import torch.nn as nn
from torch.nn import functional as F
from mine.datasets import load_dataloader
from mine.models.mine import Mine
from mine.utils.helpers import AverageMeter
from tqdm import tqdm
import numpy as np
import random

import pytorch_lightning as pl
from pytorch_lightning import Trainer


def batch_misclass_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


def batch_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)


class IBNetwork(pl.LightningModule):
    def __init__(self,
                 input_dim,
                 K,
                 output_dim,
                 mi_estimator,
                 lr,
                 beta=1.0):

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * K)
        )

        self.decoder = nn.Linear(K, output_dim)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax()

        self.mi_estimator = mi_estimator

        self.ce_loss_fn = nn.CrossEntropyLoss()

        self.lr = lr
        self.beta = beta

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        if self.on_gpu:
            x = x.cuda()

        mu, log_sigma = self.layers(x).chunk(2, dim=1)
        sigma = self.softplus(log_sigma)

        noise = torch.randn_like(sigma)
        if self.on_gpu:
            noise = noise.cuda()

        z = mu + sigma * noise

        return z, self.decoder(z)

    def configure_optimizers(self):
        opt_ib = torch.optim.Adam(
            itertools.chain(self.parameters(), self.mi_estimator.parameters()), lr=self.lr)
        return opt_ib

    def loss_fn(self, img, label):
        # Flatten
        img = img.view(img.shape[0], -1)
        if self.on_gpu:
            img = img.cuda()
            label = label.cuda()

        encoded, decoded = self.forward(img)
        ce_loss = self.ce_loss_fn(decoded, label)
        mi_loss = self.mi_estimator(img, encoded)

        loss = ce_loss + self.beta * mi_loss
        return decoded, loss

    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.on_gpu:
            img = img.cuda()
            label = label.cuda()
        decoded, loss = self.loss_fn(img, label)

        accuracy, misclass_rate = self.get_stats(decoded, label)

        tensorboard_logs = {
            'loss': loss,
            'train_accuracy': accuracy,
            'train_error_rate': misclass_rate
        }

        tqdm_dict = {'accuracy': accuracy, 'error_rate': misclass_rate}

        return {
            **tensorboard_logs,
            'log': tensorboard_logs,
            'progress_bar': tqdm_dict
        }

    def get_stats(self, decoded, labels):
        preds = torch.argmax(decoded, 1).cpu().detach().numpy()
        accuracy = batch_accuracy(preds, labels.cpu().detach().numpy())
        misclass_rate = batch_misclass_rate(
            preds, labels.cpu().detach().numpy())

        return accuracy, misclass_rate

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoded, loss = self.loss_fn(x, y)
        accuracy, misclass_rate = self.get_stats(decoded, y)

        tensorboard_logs = {'val_loss': loss,
                            'val_accuracy': accuracy,
                            'val_error_rate': misclass_rate}

        tqdm_dict = {
            'val_accuracy': accuracy
        }

        return {**tensorboard_logs, 'log': tensorboard_logs, 'progress_bar': tqdm_dict}

    def validation_end(self, outputs):
        avg_acc = np.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_accuracy': avg_acc, 'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'avg_val_accuracy': avg_acc, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        decoded, loss = self.loss_fn(x, y)
        accuracy, misclass_rate = self.get_stats(decoded, y)

        tensorboard_logs = {'test_loss': loss,
                            'test_accuracy': accuracy,
                            'test_error_rate': misclass_rate}

        return {**tensorboard_logs, 'log': tensorboard_logs}

    def test_end(self, outputs):
        avg_acc = np.stack([x['test_accuracy'] for x in outputs]).mean()
        tensorboard_logs = {'test_accuracy': avg_acc}
        return {'avg_test_accuracy': avg_acc, 'log': tensorboard_logs}

    @pl.data_loader
    def train_dataloader(self):
        return load_dataloader('mnist', 256, train=True)

    @pl.data_loader
    def val_dataloader(self):
        return load_dataloader('mnist', 256, train=False)


class GaussianLayer(nn.Module):
    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, x):
        return x + self.std * torch.randn_like(x).to(self.device)


class StatisticsNetwork(nn.Module):
    def __init__(self, x_dim, z_dim, device):
        super().__init__()
        self.layers = nn.Sequential(
            GaussianLayer(std=0.3, device=device),
            nn.Linear(x_dim + z_dim, 512),
            nn.ELU(),
            GaussianLayer(std=0.5, device=device),
            nn.Linear(512, 512),
            nn.ELU(),
            GaussianLayer(std=0.5, device=device),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.layers(x)


class TishbyNet(nn.Module):
    def __init__(self, input_dim, output_dim, activation='tanh', device='cpu'):
        super().__init__()
        self.device = device

        self.fc1 = nn.Linear(input_dim, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 7)
        self.fc4 = nn.Linear(7, 5)
        self.fc5 = nn.Linear(5, 4)
        self.fc6 = nn.Linear(4, 3)
        self.fc7 = nn.Linear(3, output_dim)

        self.activation = activation
        self.softmax = nn.Softmax()

    def non_linear(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'relu':
            return F.relu(x)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.get_layer_outputs(x)[-1]

    def get_layer_outputs(self, x):
        x1 = self.non_linear(self.fc1(x))
        x2 = self.non_linear(self.fc2(x1))
        x3 = self.non_linear(self.fc3(x2))
        x4 = self.non_linear(self.fc4(x3))
        x5 = self.non_linear(self.fc5(x4))
        x6 = self.non_linear(self.fc6(x5))
        out = self.fc7(x6)
        return [x1, x2, x3, x4, x5, x6, out]

    def estimate_layerwise_mutual_information(self, x, target, iters):
        n, input_dim = target.shape
        layer_outputs = self.get_layer_outputs(x)
        layer_outputs[-1] = F.softmax(layer_outputs[-1])
        to_return = dict()
        for layer_id, layer_output in enumerate(layer_outputs):

            _, layer_dim = layer_output.shape

            statistics_network = nn.Sequential(
                nn.Linear(input_dim + layer_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 1)
            )

            mi_estimator = Mine(T=statistics_network).to(self.device)
            mi = mi_estimator.optimize(
                target, layer_output.detach(), iters=iters, batch_size=n // 1, opt=None)

            to_return[layer_id] = mi.item()
        return to_return

    def calculate_information_plane(self, x, y, iters=100):
        info_x_t = self.estimate_layerwise_mutual_information(x, x, iters)
        info_y_t = self.estimate_layerwise_mutual_information(x, y, iters)

        return info_x_t, info_y_t


def generate_samples(n_samples):
    """
    Cred: https://github.com/stevenliuyi/information-bottleneck/blob/master/information_bottleneck.ipynb
    """
    groups = np.append(np.zeros(8), np.ones(8))
    np.random.shuffle(groups)

    x_data = np.zeros((n_samples, 10))  # inputs
    x_int = np.zeros(n_samples)  # integers representing the inputs
    y_data = np.zeros((n_samples, 2))  # outputs

    for i in range(n_samples):
        random_int = random.randint(0, 1023)
        x_data[i, :] = [int(b)
                        for b in list("{0:b}".format(random_int).zfill(10))]
        x_int[i] = random_int
        y_data[i, 0] = groups[random_int % 16]
        y_data[i, 1] = 1 - y_data[i, 0]

    return x_data, y_data, x_int


def main(device):
    epochs = 1000
    K = 256
    beta = 1e-3
    lr = 1e-3
    x_dim = 28*28
    z_dim = K

    num_gpus = 1 if device == 'cuda' else 0

    t = StatisticsNetwork(x_dim, z_dim, device=device).to(device)
    mi_estimator = Mine(t, loss='mine').to(device)
    ibnetwork = IBNetwork(input_dim=28*28, K=K, output_dim=10,
                          mi_estimator=mi_estimator, lr=lr, beta=beta).to(device)

    trainer = Trainer(amp_level='02', max_epochs=epochs,
                      gpus=num_gpus, early_stop_callback=True)
    trainer.fit(ibnetwork)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(device)

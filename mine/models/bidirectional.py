import itertools
import torch
import torch.nn as nn
import numpy as np
import abc

from datasets import load_dataloader
from mine.models.mine import T, Mine
from mine.models.gan import LinearGenerator, LinearDiscriminator
import pytorch_lightning as pl
from pytorch_lightning import Trainer


class Encoder(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(img_dim, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, latent_dim))

    def forward(self, x):
        return self.layers(x)


class BiGANDiscriminator(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(img_dim + latent_dim, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 1),
            nn.Sigmoid())

    def forward(self, x, z):
        xz = torch.cat((x, z), dim=1)
        return self.layers(xz)


class BiGAN(pl.LightningModule):
    def __init__(self, latent_dim, img_dim, mi_estimator, lr, beta=0, device='cpu', dataset='gaussians'):
        super().__init__()
        self.device = device
        self.dataset = dataset

        self.latent_dim = latent_dim

        self.generator = LinearGenerator(latent_dim, img_dim).to(device)
        self.discriminator = BiGANDiscriminator(img_dim, latent_dim)
        self.encoder = Encoder(img_dim, latent_dim)
        self.mi_estimator = mi_estimator

        self.beta = beta
        self.lr = lr

        self.loss = nn.BCELoss()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(itertools.chain(self.generator.parameters(
        ), self.mi_estimator.parameters(), self.encoder.parameters()), lr=self.lr)

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        return [opt_g, opt_d]

    def sample_z(self, N):
        return torch.randn((N, self.latent_dim)).to(self.device)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):

        x_real, _ = batch

        if len(x_real.shape) > 2:
            x_real = x_real.view(x_real.shape[0], -1)

        if self.on_gpu:
            x_real = x_real.cuda()

        valid = torch.ones((x_real.shape[0], 1)).to(self.device)
        fake = torch.zeros((x_real.shape[0], 1)).to(self.device)

        if optimizer_idx == 0:
            self.z = self.sample_z(x_real.shape[0])
            # Generator
            self.generated = self.generator(self.z)
            generator_loss = self.loss(
                self.discriminator(self.z, self.generated), valid)

            # Encoder
            self.encoded = self.encoder(x_real)
            encoder_loss = self.loss(
                self.discriminator(self.encoded, x_real), fake)

            # MI
            mi_loss = self.mi_estimator(x_real, self.encoded)

            loss = generator_loss + encoder_loss + self.beta * mi_loss

            tqdm_dict = {
                'g_loss': loss
            }

        # Discriminator
        if optimizer_idx == 1:
            discriminator_loss_real = self.loss(
                self.discriminator(self.encoded.detach(), x_real), valid)
            discriminator_loss_fake = self.loss(
                self.discriminator(self.z, self.generated.detach()), fake)

            loss = .5 * \
                (discriminator_loss_fake + discriminator_loss_real)

            tqdm_dict = {
                'd_loss': loss
            }

        output = {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

        return output

    @pl.data_loader
    def train_dataloader(self):
        return load_dataloader(self.dataset, 256, train=True)

    # @pl.data_loader
    # def val_dataloader(self):
    #     return load_dataloader('mnist', 256, train=False)


def main(device):
    dataset = 'gaussians'

    if dataset == 'gaussians':
        x_dim = 2
    elif dataset == 'mnist':
        x_dim = 28*28

    z_dim = 100

    epochs = 100
    lr = 1e-4
    beta = 1.0

    statistics_network = T(x_dim, z_dim)
    mi_estimator = Mine(statistics_network)

    model = BiGAN(latent_dim=z_dim, img_dim=x_dim,
                  mi_estimator=mi_estimator, lr=lr, beta=beta, device=device, dataset=dataset)

    trainer = Trainer(max_epochs=epochs, gpus=1, early_stop_callback=False)
    trainer.fit(model)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(device)

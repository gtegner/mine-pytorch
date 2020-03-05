import torch
import torch.nn as nn
import numpy as np
import itertools

from mine.models.layers import LinearDiscriminator, LinearGenerator, DCGanDiscriminator, DCGanGenerator
from mine.models.adaptive_gradient_clipping import adaptive_gradient_clipping_

import pytorch_lightning as pl
import torchvision
import random
import matplotlib.pyplot as plt

from mine.datasets import to_onehot


class GAN(pl.LightningModule):
    def __init__(self, input_dim, output_dim,
                 discriminator_type='linear',
                 generator_type='linear',
                 conditional_dim=0,
                 mi_estimator=None,
                 device='cpu',
                 **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.device = device

        self.generator_type = generator_type

        if generator_type == 'linear':
            self.generator = LinearGenerator(
                input_dim + conditional_dim, output_dim).to(device)
        elif generator_type == 'veegan':
            self.generator = DCGanGenerator(
                latent_dim=input_dim + conditional_dim
            ).to(device)

        if discriminator_type == 'linear':
            self.discriminator = LinearDiscriminator(output_dim).to(device)
        elif discriminator_type == 'veegan':
            self.discriminator = DCGanDiscriminator().to(device)

        self.loss = nn.BCELoss()

        self.mi_estimator = mi_estimator

        self.beta = kwargs['beta']
        self.train_loader = kwargs['train_loader']
        self.lr = kwargs['lr']

        self.generated = None
        self.conditional_dim = conditional_dim

        self.smoothing = kwargs.get('smoothing')
        self.condition_on_labels = kwargs.get('condition_on_labels')
        self.condition_on_z = kwargs.get('condition_on_z')

        self.kwargs = kwargs

    # Samples from  N(0, I) of dim : input_dim

    def sample_z(self, N, conditional):
        if self.generator_type == 'linear' or self.generator_type == 'veegan':
            z = torch.rand((N, self.input_dim)).to(self.device) * 2 - 1

            if conditional is not None:
                conditional = conditional.to(self.device)
                if len(conditional.shape) < 2:
                    conditional = conditional.unsqueeze(0)

                if len(conditional) < len(z):
                    conditional = conditional.repeat(z.shape[0], 1)
                z = torch.cat((z, conditional), dim=1)
            return z
        else:
            print("Generator type must be one of 'linear' 'veegan'")

    @staticmethod
    def random_conditional(N, conditional_dim):
        random_labels = torch.randint(0, conditional_dim, size=(N,))
        conditionals = to_onehot(random_labels, N)

        return conditionals

    def plot_grid(self, samples, conditional=None):
        plt.figure()
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().data.numpy()
        if isinstance(conditional, torch.Tensor):
            conditional = conditional.cpu().data.numpy()

        if conditional is not None:
            c = np.argmax(conditional, 1)
        else:
            c = 'blue'
        plt.scatter(samples[:, 0], samples[:, 1], c=c)
        plt.suptitle('Conditional samples')

    def generate_img_grid(self, num_samples=25):
        conditional = self.random_conditional(
            num_samples, self.conditional_dim)
        z = self.sample_z(num_samples, conditional)
        generated = self.generator(z)  # num_samples x dim
        return generated, conditional

    def forward(self, N, conditional):
        z = self.sample_z(N, conditional)
        generated = self.generator(z)
        return generated

    def sample(self, N, conditional=None):
        return self.forward(N, conditional)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(itertools.chain(
            self.generator.parameters(), self.mi_estimator.parameters()), lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]

    def mi_input(self, generated, conditional, z):
        if self.kwargs.get('condition_on_labels') and conditional is not None:
            X = conditional
        elif self.kwargs.get('condition_on_z'):
            X = z
        else:
            X = None

        return X, generated

    def training_step(self, batch, batch_idx, optimizer_idx):
        if len(batch) == 2:
            x_real, conditional = batch
        else:
            x_real = batch
            conditional = None

        if not self.kwargs['use_conditional']:
            conditional = None

        if self.on_gpu:
            x_real = x_real.float().cuda()
            if conditional is not None:
                conditional = conditional.float().cuda()

        valid = torch.ones((x_real.shape[0], 1)).to(self.device)
        fake = torch.zeros((x_real.shape[0], 1)).to(self.device)

        g_loss = 0
        d_loss = 0

        if optimizer_idx == 0:
            # Generator
            z = self.sample_z(x_real.shape[0], conditional)
            self.generated = self.generator(z)
            generated_disc = self.discriminator(self.generated)

            conditional, generated = self.mi_input(
                self.generated, conditional, z)

            if conditional is not None:
                mi = self.mi_estimator(generated, conditional)
            else:
                mi = 0

            generator_loss = self.loss(generated_disc, valid)
            g_loss = generator_loss + self.beta * mi

            adaptive_gradient_clipping_(self.generator, self.mi_estimator)

            tqdm_dict = {
                'g_loss': g_loss
            }

            output = {
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }

        elif optimizer_idx == 1:
            # Discriminator
            disc_real = self.discriminator(x_real)
            if self.smoothing:
                valid = valid - 0.3*torch.rand(valid.shape).to(self.device)
            loss_real = self.loss(disc_real, valid)

            disc_fake = self.discriminator(self.generated.detach())
            loss_fake = self.loss(disc_fake, fake)

            d_loss = 0.5 * (loss_real + loss_fake)

            tqdm_dict = {
                'd_loss': d_loss
            }

            output = {
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }

        return output

    def plot_img(self, batch, batch_idx):
        x, c = batch
        z = self.sample_z(1, c[0:1])

        generated = self.generator(z)
        plt.figure()
        plt.imshow(generated[0].cpu().data.numpy())

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

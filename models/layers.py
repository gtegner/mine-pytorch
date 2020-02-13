
import torch
import torch.nn as nn
from torch.nn import functional as F


class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)


class CustomSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class MNISTClassifier(nn.Module):
    def __init__(self, num_classes=10):
        self.layers = nn.Sequential(
            ConvBlock(3, 16, kernel_size=5, stride=1, padding=2),
            ConvBlock(16, 32, kernel_size=5, stride=1, padding=2),
            ConvBlock(32, 64, kernel_size=5, stride=2, padding=2),
            ConvBlock(64, 128, kernel_size=5, stride=2, padding=2),
        )

        self.fc1 = nn.Linear(64 * 4 * 4, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        h = self.layers(x)
        return torch.softmax(self.fc1(x))

    def loss_fn(self, x, y):
        out = self.forward(x)
        loss = self.loss(out, y)
        return loss


class LinearGenerator(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.LeakyReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, output_dim))

    def forward(self, x):
        return self.layers(x)


class LinearDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 1),
            nn.Sigmoid())

        self.input_dim = input_dim

    def forward(self, x):
        if(len(x.shape) != 2):
            x = x.view(x.shape[0], -1)

        return self.layers(x)


class DCGanGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 2 * 2 * 512)
        self.conv1 = nn.ConvTranspose2d(
            512, 256, kernel_size=5, stride=1, padding=1, output_padding=0)
        self.conv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv4 = nn.ConvTranspose2d(
            64, 3, kernel_size=5, stride=2, padding=3)

    def forward(self, input):
        x = self.fc1(input)
        x = x.view(x.size(0), 512, 2, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return torch.tanh(x)


class DCGanDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)

        self.fc1 = nn.Linear(2 * 2 * 512, 1)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        # Flatten
        x = x.view(x.size(0), -1)

        return F.sigmoid(self.fc1(x))


class ConvolutionalStatisticsNetwork(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5,
                               stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5,
                               stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=2, padding=2, bias=False)

        self.fc1 = nn.Linear(4 * 4 * 64, 1)

        self.z_linear1 = nn.Linear(z_dim, 16)
        self.z_linear2 = nn.Linear(z_dim, 32)
        self.z_linear3 = nn.Linear(z_dim, 64)

    def xz_block(self, x, z, x_layer, z_layer):
        x_out = x_layer(x)
        z_map = z_layer(z).unsqueeze(-1).unsqueeze(-1).expand_as(x_out)
        return F.elu(x_out + z_map)

    def forward(self, x, z):
        x = self.xz_block(x, z, self.conv1, self.z_linear1)
        x = self.xz_block(x, z, self.conv2, self.z_linear2)
        x = self.xz_block(x, z, self.conv3, self.z_linear3)

        x = x.view(x.size(0), -1)
        return self.fc1(x)

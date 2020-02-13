import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import MultivariateNormal


def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))


def two_spirals_torch(n, noise=.5):
    c1, c2 = twospirals(n, noise)
    return torch.from_numpy(c1), torch.from_numpy(c2)


def to_onehot(labels, num_points):
    num_classes = len(torch.unique(labels))
    one_hot = torch.zeros((num_points, num_classes))

    one_hot[torch.arange(num_points), labels.long()] = 1.0

    return one_hot


def to_onehot2(label, num_classes):

    if isinstance(label, int):
        num_points = 1
    elif len(label.shape) == 2:
        num_points = label.shape[0]
    else:
        num_points = 1

    one_hot = torch.zeros((num_points, num_classes))
    one_hot[torch.arange(num_points), label] = 1.0

    return one_hot


class FunctionDataset(torch.utils.data.Dataset):
    def __init__(self, N, dim, sigma, f):
        self.X = torch.rand((N, dim)) * 2 - 1
        self.Y = f(self.X) + torch.randn_like(self.X) * sigma

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MultivariateNormalDataset(torch.utils.data.Dataset):
    def __init__(self, N, dim, rho):
        self.N = N
        self.rho = rho
        self.dim = dim

        self.dist = self.build_dist
        self.x = self.dist.sample((N, ))
        self.dim = dim

    def __getitem__(self, ix):
        a, b = self.x[ix, 0:self.dim], self.x[ix, self.dim:2 * self.dim]
        return a, b

    def __len__(self):
        return self.N

    @property
    def build_dist(self):
        mu = torch.zeros(2 * self.dim)
        dist = MultivariateNormal(mu, self.cov_matrix)
        return dist

    @property
    def cov_matrix(self):
        cov = torch.zeros((2 * self.dim, 2 * self.dim))
        cov[torch.arange(self.dim), torch.arange(
            start=self.dim, end=2 * self.dim)] = self.rho
        cov[torch.arange(start=self.dim, end=2 * self.dim),
            torch.arange(self.dim)] = self.rho
        cov[torch.arange(2 * self.dim), torch.arange(2 * self.dim)] = 1.0

        return cov

    @property
    def true_mi(self):
        return -0.5 * np.log(np.linalg.det(self.cov_matrix.data.numpy()))


class Spirals(torch.utils.data.Dataset):
    def __init__(self, n_points, noise=.5):
        spirals, labels = twospirals(n_points, noise)

        self.num_points = len(spirals)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        spirals = self.scaler.fit_transform(spirals)

        labels = torch.from_numpy(labels)

        self.labels = to_onehot(labels, self.num_points)
        self.spirals = torch.from_numpy(spirals)

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return self.spirals[idx], self.labels[idx]


class Gaussians(torch.utils.data.Dataset):
    def __init__(self, n_points, std=1.0):
        self.n = n_points
        distance = 3 * 1.41
        centers = []
        for i in range(5):
            for j in range(5):
                center = [i * distance, j * distance]
                centers.append(center)

        x = []
        labels = []

        for ix, center in enumerate(centers):
            rand_n = np.random.multivariate_normal(
                center, np.eye(2) * std**2, size=n_points)
            label = np.ones(n_points) * ix
            x.extend(rand_n)
            labels.extend(label)

        x = np.asarray(x)
        self.labels = np.asarray(labels)

        # normalize
        self.x_np = (x - np.mean(x)) / np.std(x)

        self.x = torch.from_numpy(self.x_np).float()
        self.labels_onehot = to_onehot(
            torch.from_numpy(self.labels), len(self.labels)).float()

    def plot(self):
        plt.scatter(self.x_np[:, 0], self.x_np[:, 1],
                    c=self.labels.cpu().detach().numpy())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels_onehot[idx]


def load_dataloader(name, batch_size, train=True, N=5000):

    if name == 'mnist':
        kwargs = {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.5], [0.5])
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        return train_loader

    elif name == 'spiral':
        train_loader = torch.utils.data.DataLoader(
            Spirals(n_points=N), batch_size=batch_size, shuffle=True
        )

    elif name == 'gaussians':
        train_loader = torch.utils.data.DataLoader(
            Gaussians(n_points=N // 25, std=1.0), batch_size=batch_size, shuffle=True
        )

    return train_loader


class StackedMNIST(torch.utils.data.Dataset):
    def __init__(self, N, train=True, seed=42, device='cpu', noise_std=0.1):
        np.random.seed(seed)
        self.IMG_WIDTH, self.IMG_HEIGHT = 28, 28
        self.mnist = datasets.MNIST('../data', train=train, download=True,
                                    transform=transforms.Compose([
                                        transforms.Scale(self.IMG_WIDTH),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])
                                    ]))

        self.n = len(self.mnist)
        self.N = N
        self.device = device
        self.noise_std = noise_std

    def __len__(self):
        return self.N or len(self.mnist)**3

    def get_random_img(self):
        img, label = self.mnist[np.random.randint(low=0, high=self.n)]
        label_onehot = to_onehot2(label, num_classes=10)

        img = self.add_noise(img)

        return img, label_onehot

    def add_noise(self, img):
        return img + torch.randn_like(img)*self.noise_std

    def __getitem__(self, idx):
        images = torch.zeros((3, self.IMG_WIDTH, self.IMG_HEIGHT))
        labels = np.array([])

        for i in range(3):
            img, label = self.get_random_img()
            labels = np.append(labels, label)
            images[i] = img[0]

        return images, torch.from_numpy(labels).float()

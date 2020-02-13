import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
PLOT_DIR = 'figures'

if not os.path.exists(PLOT_DIR):
    os.mkdir(PLOT_DIR)


def batch(x, y, batch_size=1, shuffle=True):
    assert len(x) == len(
        y), "Input and target data must contain same number of elements"
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        y = y[rand_perm]

    batches = []
    for i in range(n // batch_size):
        x_b = x[i * batch_size: (i + 1) * batch_size]
        y_b = y[i * batch_size: (i + 1) * batch_size]

        batches.append((x_b, y_b))
    return batches


class AverageMeter:
    def __init__(self, name='', percentage=False):
        self.mu = 0
        self.n = 0
        self.mus = []
        self.name = name
        self.percentage = percentage

    def __add__(self, total_mean, mu, x):
        n = len(total_mean)
        mu = (mu * n + x) / (n + 1)
        total_mean.append(mu)

        return total_mean

    def add(self, x):
        total_mean = self.__add__(self.mus, self.mu, x)
        self.mu = total_mean[-1]
        self.mus = total_mean
        self.n = len(self.mus)

    def reset(self):
        self.mu = 0
        self.n = 0
        self.mus = []

    def plot(self, save=False):
        plt.figure()
        plt.scatter(np.arange(self.n), self.mus)
        if save:
            plt.savefig(f"{PLOT_DIR}/{self.name}_run_{self.n}.png")
        plt.close()

    def __str__(self):
        if self.percentage:
            mu = self.mu * 100
        else:
            mu = self.mu
        return "{} average: {:3g}".format(self.name, mu)

    @property
    def value(self):
        return self.mu

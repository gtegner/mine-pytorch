from mine.models.bidirectional import BiGAN
from mine.models.information_bottleneck import IBNetwork
from mine.models.gan import GAN
from mine.models.mine import T, EnergyLoss

import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs')
    parser.add_argument('--model')
    parser.add_argument('--lr')
    parser.add_argument('--batch-size')
    parser.add_argument('--dataset')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    main(args)

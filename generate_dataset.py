import argparse
import numpy as np
from dataset.dataset_generation import DatasetGeneration


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', default=1,
                    help='The linear equation and the centre of samples \
                    will be affected by this seed.')
parser.add_argument('--size', default=int(1000*10/6),
                    help='Size of dataset in total.')
parser.add_argument('-S', '--savedir', default='data',
                    help='Dir for saving datasets.')
args = parser.parse_args()


if __name__ == '__main__':
    np.random.seed(args.seed)
    # make sure the `size` is even.
    args.size += args.size % 2

    dataset = DatasetGeneration(args.size)
    dataset.visualization()
    dataset.save(args.savedir)

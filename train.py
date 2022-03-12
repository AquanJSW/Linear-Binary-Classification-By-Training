from itertools import product
from tqdm import tqdm
from copy import deepcopy
from os import makedirs
import os
from benchmark.evaluate import evaluate
from utils.utils import onehot2scalar
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from model.Linear import Linear
import torch
import torch.nn.functional as F
import dataset.dataset as dataset
import argparse


class config:
    epoch = 10
    batch_sizes = (4,)
    shuffle = True
    error_r = 0.01  # stop training when error rate is lower than this on valset.
    lrS = (1e-3,)
    keys_result_on_fold = ['loss', 'error', 'state_dict']


parser = argparse.ArgumentParser()
parser.add_argument(
    '--datapath', default='data/dataset_train.bin', help='Dataset path.'
)
parser.add_argument('--modelpath', default='checkpoints/bestmodel.pts')
# parser.add_argument('-e', '--epoch', default=10)
args = parser.parse_args()

datasets = dataset.load_dataset(args.datapath)['dataset']

model = Linear()

writer = SummaryWriter()


def get_loss(output, target):
    """

    :param output: ndarray of shape (batch_size, 1)
    :param target: ndarray of shape (batch_size, 2), which are one-hots
    """
    gt = torch.tensor([onehot2scalar(x) for x in target], dtype=torch.float32).reshape(
        len(target), 1
    )
    return F.binary_cross_entropy(output, gt)


# def init_module(m: torch.nn.Module):
#     if type(m) == torch.nn.Linear:
#         m.weight.fill_(.0)
#         m.bias = 0
#     print(m.weight)
#     print(m.bias)


def train(
    trainset: np.ndarray,
    valset: np.ndarray,
    batch_size,
    optimizer,
    writer_log_hier_base,
):
    """Final train function in fold, on given datasets.

    Args
    ---
    `trainset`: Shape (cSample, 2, 2).
    `valset`: Same as above.

    Return
    ---
    loss, error, state_dict
    """
    trainset_ = dataset.Dataset(trainset)

    trainsetloader = DataLoader(
        trainset_, batch_size=batch_size, shuffle=config.shuffle
    )
    model.train()
    for i_e in range(config.epoch):
        for i_s, (x, y) in enumerate(trainsetloader):
            model.zero_grad()
            i = i_e * len(trainsetloader) + i_s
            x.requires_grad = True
            output = model(x)
            loss = get_loss(output, y)
            writer.add_scalar(writer_log_hier_base + 'Loss/train', loss, i)
            weight = model.get_parameter('_linear.weight').flatten()
            bias = model.get_parameter('_linear.bias')
            writer.add_scalar(writer_log_hier_base + 'Param/weight[0]', weight[0], i)
            writer.add_scalar(writer_log_hier_base + 'Param/weight[1]', weight[1], i)
            writer.add_scalar(writer_log_hier_base + 'Param/bias', bias[0], i)
            loss.backward()
            optimizer.step()
        model.eval()
        val_error = 1 - np.min(evaluate(model, valset))
        model.train()
        writer.add_scalar(writer_log_hier_base + 'Error/val', val_error, i_e)
        # if val_error < config.error_r:
        #     return loss, val_error, deepcopy(model.state_dict())
    return loss, val_error, deepcopy(model.state_dict())


def train_on_folds(batch_size, lr, writer_log_hier_base):
    """Have a train with given hyperparameters.

    Return
    ---
    The result of fold with minimum error.
    """
    results = []
    for foldn, dataset_ in enumerate(datasets):
        writer_log_hier = writer_log_hier_base + 'fold{}/'.format(foldn)
        global model
        model = Linear()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        values_result = train(
            dataset_['trainset'].astype(np.float32),
            dataset_['valset'].astype(np.float32),
            batch_size,
            optimizer,
            writer_log_hier,
        )
        result = dict(zip(config.keys_result_on_fold, values_result))
        results.append(result)
    errors = [result['error'] for result in results]
    idx_min_error = errors.index(min(errors))
    writer.add_text(
        writer_log_hier_base + 'Result',
        'Best fold {} with error {}'.format(idx_min_error, errors[idx_min_error]),
    )
    return results[idx_min_error]


def train_on_hyperparameters() -> dict:
    """

    Return
    ---
    A dict of (batch_size, lr, loss, error, state_dcit) with minimum error.
    """
    keys_hyperparameters = ['batch_size', 'lr']
    results = []
    values_hyperparameters = []
    for batch_size, lr in tqdm(product(config.batch_sizes, config.lrS)):
        values_hyperparameters.append((batch_size, lr))
        print("batch_size {} lr {}".format(batch_size, lr))
        writer_log_hier = 'bs{}_lr{}_ep{}/'.format(batch_size, lr, config.epoch)
        result = train_on_folds(batch_size, lr, writer_log_hier)
        results.append(result)

    hyperparameters = [
        dict(zip(keys_hyperparameters, vals)) for vals in values_hyperparameters
    ]
    errors = [result['error'] for result in results]
    idx_min_error = errors.index(min(errors))
    writer.add_text(
        'Result',
        'best hyperpararmeters:\n{}\nwith error {}'.format(
            hyperparameters[idx_min_error], errors[idx_min_error]
        ),
    )
    results[idx_min_error].update(hyperparameters[idx_min_error])
    return results[idx_min_error]


def main():
    # writer.log_dir = 'runs/' + time.strftime('%y%m%d_%H-%M-%S_') + os.uname()[1] + '/'
    result = train_on_hyperparameters()

    makedirs(os.path.dirname(args.modelpath), exist_ok=True)
    torch.save(result['state_dict'], args.modelpath)
    print("model is saved to {}".format(args.modelpath))
    return 0


if __name__ == '__main__':
    exit(main())

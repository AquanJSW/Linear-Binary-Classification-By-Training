import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset import Dataset
from .FMeasure import FMeasure
from utils.utils import onehot2scalar
import numpy as np

_batch_size = 16


def evaluate(model: nn.Module, dataset: np.ndarray):
    """Evaluate model on given dataset.

    Note that invoking `model.eval()` before invoking this function.

    Args
    ---
    `model`: Model in eval mode.
    `dataset`: Shape (cSample, 2, 2)

    Return
    ---
    All the classes' F-Measure.
    """
    fmeasure = FMeasure(2)
    dataset_ = Dataset(dataset)
    dataloader = DataLoader(dataset_, _batch_size)
    for samples, labels in dataloader:
        samples.requires_grad = False
        labels.requires_grad = False
        predicted = np.round(model(samples).detach().numpy()).astype(int)
        groundtruth = [onehot2scalar(x) for x in labels.detach().numpy()]
        [fmeasure.update(p, g) for p, g in zip(predicted, groundtruth)]
    return fmeasure.get_result()

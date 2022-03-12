import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from termcolor import colored
from .linear_equation import LinearEquation
import sys

sys.path.append(sys.path[0] + '/../')  # nopep8
from utils.utils import randfloat  # nopep8

# Linear equation's coefficient `a` and `b`, together with the x coordination
# of the samples' centre shall be randomly generated from this interval.
# linear function: ax+y+b = 0
_rand_interval_a = [-2, 2]
_rand_interval_b = [-1, 1]
# The centre of samples.
_x0 = 5
# x and y coordinate shift w.r.t. the centre of samples.
_rand_interval_shift = 5
# trainset, valset and testset's ratio w.r.t. total.
_train_val_test_ratio = (0.6, 0.2, 0.2)
# k-fold in cross-validation
_k = 10


def get_linear_equation():
    a = randfloat(*_rand_interval_a)
    b = 1
    c = randfloat(*_rand_interval_b)
    return LinearEquation(a, b, c)


def _normalize_ratio(ratio):
    return np.array(ratio) / sum(ratio)


class DatasetGeneration:
    def __init__(self, size) -> None:
        self._linear_equation = get_linear_equation()
        self._class_0 = np.array([1, 0], np.float32)
        self._class_1 = np.array([0, 1], np.float32)
        self._size = size
        self._centre = self._get_centre()
        self._samples = self._get_samples()

    def _get_centre(self) -> np.ndarray:
        # dataset's centre point
        # x0 = np.random.randint(*_rand_interval_abx)
        # without loss of generality.
        y0 = self._linear_equation.solve_y(_x0)
        centre = np.array([_x0, y0], dtype=np.float32)
        return centre

    def _get_label(self, point):
        return self._class_0 if self._linear_equation.get_sign(*point) else self._class_1

    def _get_samples(self):
        """
        Return full dataset.

        Note that the first half of `return` belongs to one class, and the
        left belongs to the other class.

        :return shape (sample_n, 2, 2)
        """

        ret = np.ndarray([0, 2, 2], np.float32)
        c_0_c = 0  # class 0 count
        c_1_c = 0
        try_c = 0  # try count
        while c_0_c + c_1_c != self._size:

            point = (
                np.array(
                    [
                        randfloat(-_rand_interval_shift, _rand_interval_shift),
                        randfloat(-_rand_interval_shift, _rand_interval_shift),
                    ],
                    np.float32,
                )
                + self._centre
            )
            label = self._get_label(point)
            sample = np.array([point, label])
            try_c += 1

            def is_dup():
                """Is sample in ret?"""
                result = sample[0] == ret[:, 0]
                return np.any(result[:, 0] & result[:, 1])

            # skip duplication
            if is_dup():
                print(
                    colored(
                        "\rduplicated, skipping...valid: {}/{}".format(
                            c_0_c + c_1_c, try_c
                        ),
                        "red",
                    ),
                    end='',
                )
                continue

            if np.all(label == self._class_0):
                if c_0_c == self._size // 2:
                    continue
                ret = np.insert(ret, 0, sample, 0)
                c_0_c += 1
                print(
                    colored("\nclass 0: {}/{}".format(c_0_c, try_c), "yellow"), end=''
                )
            else:
                if c_1_c == self._size // 2:
                    continue
                ret = np.append(ret, sample.reshape(1, 2, 2), 0)
                c_1_c += 1
                print(colored("\nclass 1: {}/{}".format(c_1_c, try_c), "green"), end='')

        return ret

    def _get(self):
        """Get dataset for k-fold cross-validation."""
        class_pairs = np.array(
            [
                [self._samples[i], self._samples[i + self._size // 2]]
                for i in range(self._size // 2)
            ]
        )

        # fix the testset (not involve fold)
        size_dataset_test = int(
            len(class_pairs) * _normalize_ratio(_train_val_test_ratio)[2]
        )
        dataset_test = class_pairs[:size_dataset_test].reshape(-1, 2, 2)
        class_pairs = class_pairs[size_dataset_test:]
        train_val_ratio = _normalize_ratio(_train_val_test_ratio[:2])

        # get k-folds with some pairs dropped out
        folds = np.array(np.split(class_pairs[: -(len(class_pairs) % _k)], _k))

        # trainset and valset's fold count
        train_val_foldc = np.array(train_val_ratio, dtype=float) * _k
        train_val_foldc = np.floor(train_val_foldc)
        np.array(train_val_foldc, dtype=int)
        # Make sure all the folds are in use
        if sum(train_val_foldc) != _k:
            # Though it's a rare event.
            res = _k - sum(train_val_foldc)
            train_val_foldc[0] += res
        # Zero check of `foldc`? Maybe no need?
        train_val_foldc = train_val_foldc.astype(int)

        dataset_train = []
        indices = np.arange(len(folds), dtype=int)
        for i in range(_k):
            indices -= 1
            trainset_indices = indices[: train_val_foldc[0]]
            valset_indices = indices[train_val_foldc[0] : train_val_foldc[0] + train_val_foldc[1]]
            trainset = folds[np.ix_(trainset_indices)]
            valset = folds[np.ix_(valset_indices)]
            trainset = np.concatenate(
                (*trainset, class_pairs[-(len(class_pairs) % _k) :])
            ).reshape(-1, 2, 2)
            valset = np.concatenate(valset).reshape(-1, 2, 2)
            dataset_train.append({"trainset": trainset, "valset": valset})
        return dataset_train, dataset_test

    def save(self, dir):
        """Save dataset, linear equation into the specified file.
        
        Two dataset files will be dumpped: train and test.
        Each of which has the following architecture:
        ```
        dict {
            'dataset': ...,
            'a': ...,
            'b', ...,
            'c', ...
        }
        ```
        The 'a', 'b' and 'c' fields come from `ax+by+c=0`
        """
        os.makedirs(dir, exist_ok=True)
        dataset_train, dataset_test = self._get()
        data_train = {
            'dataset': dataset_train,
            'a': self._linear_equation.a,
            'b': self._linear_equation.b,
            'c': self._linear_equation.c,
        }
        data_test = {
            'dataset': dataset_test,
            'a': self._linear_equation.a,
            'b': self._linear_equation.b,
            'c': self._linear_equation.c,
        }
        path_dataset_train = os.path.join(dir, 'dataset_train.bin')
        path_dataset_test = os.path.join(dir, 'dataset_test.bin')
        with open(path_dataset_train, 'wb') as fp:
            pickle.dump(data_train, fp)
        with open(path_dataset_test, 'wb') as fp:
            pickle.dump(data_test, fp)
        print("\ntrain dataset has been saved to {}".format(path_dataset_train))
        print("test dataset has been saved to {}".format(path_dataset_test))

    def visualization(self, path='figure/figure.png'):
        # linear function
        x_line = np.linspace(
            self._centre[0] - _rand_interval_shift,
            self._centre[0] + _rand_interval_shift,
        )
        y_line = self._linear_equation.solve_y(x_line)
        # dots
        x_class_0 = self._samples[: self._size // 2, 0, 0]
        y_class_0 = self._samples[: self._size // 2, 0, 1]
        x_class_1 = self._samples[self._size // 2 :, 0, 0]
        y_class_1 = self._samples[self._size // 2 :, 0, 1]

        fig, ax = plt.subplots(1, 1)
        ax.plot(x_line, y_line, '-')
        ax.plot(x_class_0, y_class_0, 'o', color='b')
        ax.plot(x_class_1, y_class_1, 'o', color='r')

        plt.xlim(
            self._centre[0] - _rand_interval_shift * 1.1,
            self._centre[0] + _rand_interval_shift * 1.1,
        )
        plt.ylim(
            self._centre[1] - _rand_interval_shift * 1.1,
            self._centre[1] + _rand_interval_shift * 1.1,
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        print("\nfig is saved to {}".format(path))

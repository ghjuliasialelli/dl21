import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset


# TODO: Decide if we're using PyTorch & Lightning or Tensorflow & Keras
class DigitData(Dataset):

    def __init__(self, path: str, cj_variances: dict([str, int])):
        """"
        @:param:    path
        @:param:    cj_variances contains the color-jitter-variance values and corresponding expected bias (strong,
        middle, weak). For explanation, see https://github.com/feidfoe/learning-not-to-learn.
        """
        super.__init__()
        # Load the colored mnist dataset into memory
        self.path = path
        self.cj_variances = cj_variances
        self.db = dict([])

        for name in cj_variances.keys():
            self.db.update(
                [name, np.load(os.path.join(path, f'mnist_10color_jitter_var_{cj_variances[name]}.npy'),
                               encoding='latin1', allow_pickle=True).item()]
            )

    def _set_bias_(self, bias: str) -> None:
        self.cjv = self.cj_variances[bias]
        self.bias = bias

    def __getitem__(self, item):
        return self.db[self.bias][item]

    def __len__(self):
        return len(self.db[self.bias])

    def __str__(self):
        info = f' DigitDatabase with {len(self.cj_variances)} subsets.\n' \
               f' Subsets filtered by color-jitter variances (i.e. expected bias).'
        return info

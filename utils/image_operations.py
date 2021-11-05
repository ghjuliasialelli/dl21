import os
import numpy as np
import torch
from torch.distributions import transforms
from torch.utils.data import Dataset as TorchDataset
from PIL import Image

from tensorflow.data import Dataset as TfDataset

# TODO: Decide if we're using PyTorch & Lightning or Tensorflow & Keras
class DigitData_Torch(TorchDataset):

    def __init__(self, path: str, cj_variance: str, mode: str):
        """"
        @:param:    path
        @:param:    cj_variance contains the color-jitter-variance and corresponding string.
                    For more, see https://github.com/feidfoe/learning-not-to-learn.
        @:param:    mode is set to "train","test" or "test_gray".
        """
        super().__init__()
        # Load the colored mnist dataset into memory
        self.path = path
        self.cj_variance = cj_variance + '0'
        self.mode = mode

        # keys from np.load(..) are:
        # ['test_image', 'test_label', 'test_gray', 'train_label', 'train_image']
        if mode != 'test_gray':
            self.images = np.load(os.path.join(path, f'mnist_10color_jitter_var_{self.cj_variance}.npy'),
                                  encoding='latin1', allow_pickle=True).item()[self.mode + '_image']
            self.labels = np.load(os.path.join(path, f'mnist_10color_jitter_var_{self.cj_variance}.npy'),
                              encoding='latin1', allow_pickle=True).item()[self.mode + '_label']
        else:
            self.images = np.load(os.path.join(path, f'mnist_10color_jitter_var_{self.cj_variance}.npy'),
                                  encoding='latin1', allow_pickle=True).item()[self.mode]
            self.labels = None

        """self.ToPIL = transforms.Compose([
            transforms.ToPILImage(),
        ])"""

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]

        # x = self.ToPIL(x)

        return x, y

    def __len__(self):
        # use shape?
        return len(self.images)

    def __str__(self):
        info = f' DigitDatabase...\n' \
               f' Color-jitter variances {self.cj_variance}.\n' \
               f' Number of Images: {len(self.images)}\n' \
               f' Labels present: {self.labels is not None}'
        return info

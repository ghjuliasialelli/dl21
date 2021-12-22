import os
import numpy as np
import torch
from torch.distributions import transforms
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from PIL.ImageOps import *

import tensorflow as tf
from tensorflow.data import Dataset as TfDataset


class DigitData_Torch(TorchDataset):
    """
    Usage of the dataset:

    """

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

    def __getitem__(self, index):
        x = self.images[index]
        if self.mode != 'test_gray':
            y = self.labels[index]
            return x, y
        else:
            # test_gray doesnt have labels (?)
            return x

    def __len__(self):
        # use shape?
        return len(self.images)

    def __str__(self):
        info = f' DigitDatabase...\n' \
               f' Color-jitter variances {self.cj_variance}.\n' \
               f' Number of Images: {len(self.images)}\n' \
               f' Labels present: {self.labels is not None}'
        return info


class DigitData_TF:
    """
    This is just a class to retrieve a Tensorflow dataset containing the Digit Data.
    ->  By calling .get_dataset we obtain a tensorflow dataset which can be used to load data into
        a tensorflow models.
    *   Batch size and other parameters can be used when calling _load(). TO BE IMPLEMENTED.
    """
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    def __init__(self, path: str, cj_variance: str, mode: str):
        self.cj_variance = cj_variance + '0'
        self.mode = mode
        self.path = path
        self.dataset = None
        # load dataset on init?
        try:
            self._load(path=path)
        except:
            print(f"Error while loading dataset from {path}")

    def _load(self, path: str):
        data = np.load(os.path.join(path, f'mnist_10color_jitter_var_{self.cj_variance}.npy'), encoding='latin1',
                       allow_pickle=True)
        images = data.item()[self.mode + '_image']
        if self.mode != 'test_gray':
            labels = data.item()[self.mode + '_label']
            self.dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices(images)

    def get_dataset(self):
        assert self.dataset is not None, "DigitDataset has not been loaded with data.\nConsider using .load() first."
        """
        self.dataset = self.dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        _or_
        self.dataset = self.dataset.batch(BATCH_SIZE)
        """
        return self.dataset


class LucasDigitData(TorchDataset):
    """
    Usage of the dataset:

    """

    def __init__(self, device, data_files: list, train=True, test=False):
        """"
        @:param:    path
        @:param:    cj_variance contains the color-jitter-variance and corresponding string.
                    For more, see https://github.com/feidfoe/learning-not-to-learn.
        @:param:    mode is set to "train","test" or "test_gray".
        """
        super().__init__()
        # Load the colored mnist dataset into memory
        self.device = device

        self.data = []
        self.labels = []
        for file in data_files:
            np_data = np.load(file, encoding='latin1', allow_pickle=True).item()
            if train:
                self.data.append(np_data['train_image'])
                self.labels.append(np_data['train_label'])
            if test:
                self.data.append(np_data['test_image'])
                self.labels.append(np_data['test_label'])

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        # use shape?
        return len(self.data)

    def __getitem__(self, index):
        return torch.permute(torch.from_numpy(self.data[index]), (2, 0, 1)).float().to(self.device), \
               torch.from_numpy(np.array([self.labels[index]])).to(self.device)

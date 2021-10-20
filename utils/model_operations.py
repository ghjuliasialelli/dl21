import h5py
import os
import sys

from torch.utils.data import IterableDataset, Dataset
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# For conversion from tensorflow to pytorch, see:
# https://github.com/Cadene/tensorflow-model-zoo.torch/blob/master/inceptionv4/pytorch_load.py

class ModelDataset(Dataset):

    def __init__(self, bias: (str, float), data_directory: str,
                 model):
        """"
        @:param:    bias
        @:param:    model_directory
        @:param:    model should be the DigitClassifier model, as implemented in
                    get_weights.py
        """
        self.model_directory = os.path.join(data_directory, str(bias))
        self.num_models = len(os.listdir(self.model_directory))
        
        for name in os.listdir(self.model_directory):
            f = h5py.File(os.path.join(self.model_directory, name), 'r')


    def __len__(self):
        return self.num_models

    def __getitem__(self, index):
        pass

    def build_model(self):
        pass
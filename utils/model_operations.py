# ------------
# Author:       Philip Toma
# Description:  This file implements the datasets for model weights.
# ------------


import h5py
import os
import sys

from torch.utils.data import IterableDataset, Dataset, random_split, ConcatDataset
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from pandas import read_pickle, concat


# For conversion from tensorflow to pytorch, see:
# https://github.com/Cadene/tensorflow-model-zoo.torch/blob/master/inceptionv4/pytorch_load.py


def balance_datasets(train_data: Dataset, test_data: Dataset, split1: [int], split2: [int]):
    """Split order: [num_elements going to the train set, num_elements going to the test set]"""
    train_split = random_split(train_data, split1)
    test_split = random_split(test_data, split2)
    return ConcatDataset([train_split[0], test_split[0]]), ConcatDataset([train_split[1], test_split[1]])


class ModelDataset(Dataset):

    def __init__(self, bias: str, data_directory: str, new_model: bool = False):
        """"
        @:param:    bias
        @:param:    model_directory
        @:param:    models should be the DigitClassifier models, as implemented in
                    get_weights.py

        Descr.:     This dataset can be used to feed the pretrained models. Either feed this using a torch dataloader
                    to a torch models, or use ModelDataset[i] to fetch the i-th models from the directory.
                    ##For future: may want to store to disk once we have loaded a pretrained models. However, at the
                    moment this is not necessary.##
        """
        super(ModelDataset, self).__init__()
        self.model_directory = os.path.join(data_directory, str(bias))
        self.num_models = len(os.listdir(self.model_directory))
        self.bias = bias
        self.new_model = new_model

    def _build_digit_classifier(self):
        """
        Build the digit classifier models (TF) that was used in the IFBID paper. Weights are loaded as shown
        in load_model. Load_model() should be used to load a models with its weights into the dataset.
        :return: Model architecture without pre-trained weights.
        """
        model = Sequential()
        if not self.new_model:
            model.add(Conv2D(24, kernel_size=5, activation='relu', input_shape=(28, 28, 3)))
            model.add(MaxPooling2D())

            model.add(Conv2D(48, kernel_size=3, activation='relu'))
            model.add(MaxPooling2D())

            model.add(Conv2D(64, kernel_size=3, activation='relu'))
            model.add(MaxPooling2D())

            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        else:
            model.add(Conv2D(24, kernel_size=5, activation='relu', input_shape=(32, 32, 3)))
            model.add(MaxPooling2D())

            model.add(Conv2D(48, kernel_size=3, activation='relu'))
            model.add(MaxPooling2D())

            model.add(Conv2D(64, kernel_size=3, activation='relu'))
            model.add(MaxPooling2D())

            model.add(Flatten())
            model.add(Dense(3163, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def print_digit_classifier_info(self, index: int, TF_summary: bool = False):
        model = self.load_model(index)
        if TF_summary:
            print(model.summary())
        else:
            weights = model.get_weights()
            n_layers = len(weights)
            layer_info = []
            print('--- --- ---')
            for i in range(0, n_layers):
                # Print information about the shape of the weights that are at each layer.
                # May be more useful for planning than just the information given by TF.
                print(f'Layer {i} shape of the Digit Classifier: {weights[i].shape}')
                print('--- --- ---')

    def load_model(self, model_number: int):
        model = self._build_digit_classifier()
        model.load_weights(os.path.join(self.model_directory, f'{model_number}.h5'))
        return model

    def concat(self):
        pass

    def __len__(self):
        return self.num_models

    def __getitem__(self, index):
        model = self.load_model(model_number=index)
        return model

    def __str__(self):
        return f'ModelDataset of length {len(self)}'


class PhilipsModelDataset(ModelDataset):

    def __init__(self, bias: str, data_directory: str, num_classes, standardize=False,
                 new_model=False):
        super(PhilipsModelDataset, self).__init__(bias, data_directory)
        self.data_directory = data_directory
        # set the number of classes: 4 or 2
        self.num_classes = num_classes#4    # or =2
        self.sublength = None
        self.new_model = new_model

        self.standardize = standardize

    def __len__(self):
        # lenths of the subdirectories are 2000
        length = 0
        for directory in os.listdir(self.data_directory):
            length = length + len(os.listdir(os.path.join(self.data_directory, directory)))
            #print(len(os.listdir(os.path.join(self.data_directory, directory))))
            self.sublength = len(os.listdir(os.path.join(self.data_directory, directory)))
        return length

    def load_model_k(self, bias: str, model_number: int):
        model = self._build_digit_classifier()
        model.load_weights(os.path.join(self.data_directory, bias, f'{model_number}.h5'))
        return model

    def __getitem__(self, index):
        dir_index = index // self.sublength
        model_number = index % self.sublength
        biases = ['0.02', '0.03', '0.04', '0.05']
        model = self.load_model_k(bias=biases[dir_index], model_number=model_number)
        """except:
            biases = ['0.020', '0.030', '0.040', '0.050']
            model = self.load_model_k(bias=biases[dir_index], model_number=model_number)"""

        i = 0
        return_dict = {}
        for layer in model.layers:
            #print(layer)
            if layer.__class__.__name__ == 'Conv2D':
                weights = layer.get_weights()[0]
                #ws = torch.permute(torch.from_numpy(weights), (3, 2, 0, 1))
                ws = torch.from_numpy(weights).permute(3, 2, 0, 1)
                if self.standardize:
                    mean = ws.mean(dim=1, keepdim=True)
                    sd = ws.std(dim=1, keepdim=True)
                    ws = (ws - mean) / sd
                return_dict[f'layer_{i}'] = ws
                i = i + 1
            elif layer.__class__.__name__ == 'Dense':
                weights = layer.get_weights()[0]
                ws = torch.from_numpy(weights)
                return_dict[f'layer_{i}'] = ws
                i = i + 1
        zeros = np.zeros(self.num_classes)

        # check if we're only classifying 'biased/non-biased' or 'very high, high, low, very low'.
        if self.num_classes == 2:
            if dir_index == 1:
                dir_index = 0
            elif dir_index >= 2:
                dir_index = 1

        zeros[dir_index] = 1
        sample = {'model_weights': return_dict, 'bias': torch.from_numpy(zeros)}
        # print(f'Sample: {sample["bias"]}')
        return sample


class LucasModelDataset(Dataset):

    def __init__(self, device, data_files: list, use_weights=True, use_biases=False, only_conv=False):
        super(LucasModelDataset, self).__init__()
        # set the number of classes: 4 or 2
        self.use_weights = use_weights
        self.use_biases = use_biases
        self.only_conv = only_conv
        self.device = device

        self.data = None

        for file in data_files:
            if self.data is None:
                self.data = read_pickle(file)
            else:
                self.data = concat([self.data, read_pickle(file)], ignore_index=True)

    def __len__(self):
        # lenths of the subdirectories are 2000
        return len(self.data)

    def __getitem__(self, index):
        model = self.data.iloc[index]

        i = 0
        return_dict = {}
        for layer in model['layers']:
            #print(layer)
            if layer['name'] == 'Conv2D':
                weights = layer['weights']
                biases = layer['bias']
                if self.use_weights:
                    ws = torch.permute(torch.from_numpy(weights), (3, 2, 0, 1))
                    torch.from_numpy(weights).permute(3, 2, 0, 1)
                    return_dict[f'layer_{i}'] = ws.float().to(self.device)
                    i += 1
                if self.use_biases:
                    bs = torch.from_numpy(biases).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    return_dict[f'layer_{i}'] = bs.float().to(self.device)
                    i += 1
            elif layer['name'] == 'Dense' and not self.only_conv:
                weights = layer['weights']
                biases = layer['bias']
                if self.use_weights:
                    ws = torch.permute(torch.from_numpy(weights), (1, 0)).unsqueeze(2).unsqueeze(3)
                    return_dict[f'layer_{i}'] = ws.float().to(self.device)
                    i += 1
                if self.use_biases:
                    bs = torch.from_numpy(biases).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    return_dict[f'layer_{i}'] = bs.float().to(self.device)
                    i += 1

        zeros = np.zeros((4))

        for i, b in enumerate(['0.02', '0.03', '0.04', '0.05']):
            if b == model['bias']:
                zeros[i] = 1
                break

        sample = {'model_weights': return_dict, 'bias': torch.from_numpy(zeros).float().to(self.device)}
        # print(f'Sample: {sample["bias"]}')
        return sample
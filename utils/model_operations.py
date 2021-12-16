import h5py
import os
import sys

from torch.utils.data import IterableDataset, Dataset
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# For conversion from tensorflow to pytorch, see:
# https://github.com/Cadene/tensorflow-model-zoo.torch/blob/master/inceptionv4/pytorch_load.py

class ModelDataset(Dataset):

    def __init__(self, bias: str, data_directory: str,):
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

    def _build_digit_classifier(self):
        """
        Build the digit classifier models (TF) that was used in the IFBID paper. Weights are loaded as shown
        in load_model. Load_model() should be used to load a models with its weights into the dataset.
        :return: Model architecture without pre-trained weights.
        """
        model = Sequential()
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

    def __init__(self, bias: str, data_directory: str,):
        super(PhilipsModelDataset, self).__init__(bias, data_directory)
        self.data_directory = data_directory

    def __len__(self):
        # lenths of the subdirectories are 2000
        length = 0
        for directory in os.listdir(self.data_directory):
            length = length + len(os.listdir(os.path.join(self.data_directory, directory)))
        return length

    def load_model_k(self, bias: str, model_number: int):
        model = self._build_digit_classifier()
        model.load_weights(os.path.join(self.data_directory, bias, f'{model_number}.h5'))
        return model

    def __getitem__(self, index):
        dir_index = index // 2000
        model_number = index % 2000
        biases = ['0.02', '0.03', '0.04', '0.05']
        model = self.load_model_k(bias=biases[dir_index], model_number=model_number)

        i = 0
        return_dict = {}
        for layer in model.layers:
            #print(layer)
            if layer.__class__.__name__ == 'Conv2D': # or layer.__class__.__name__ == 'Dense':
                weights = layer.get_weights()[0]
                ws = torch.permute(torch.from_numpy(weights), (3, 2, 0, 1))
                return_dict[f'layer_{i}'] = ws
                i = i + 1
        zeros = np.zeros(4)
        zeros[dir_index] = 1
        sample = {'model_weights': return_dict, 'bias': torch.from_numpy(zeros)}
        # print(f'Sample: {sample["bias"]}')
        return sample


class Loading_Dataset(Dataset):

    def __init__(self, biases, datasets):
        super(Loading_Dataset, self).__init__()
        self.data = datasets
        self.lengths = []
        for dataset in datasets:
            self.lengths.append(len(dataset))
        self.biases = biases

    def __len__(self):
        pass

    def __getitem__(self, index):
        for i in range(len(self.lengths)):
            if index - self.lengths[0] < 0:
                return self.data[i][index] #, biases[i]
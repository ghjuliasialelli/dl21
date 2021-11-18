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
        self.model_directory = os.path.join(data_directory, str(bias))
        self.num_models = len(os.listdir(self.model_directory))

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

    def __len__(self):
        return self.num_models

    def __getitem__(self, index):
        model = self.load_model(model_number=index)
        return model

    def __str__(self):
        return f'ModelDataset of length {len(self)}'


# When testing:
#data = ModelDataset(bias='0.02', data_directory='/home/phil/Documents/Studium/DL/Project/train/')
#print(data[2].get_weights())

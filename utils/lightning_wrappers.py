# ------------
# Author:       Philip Toma
# Description:  This file implements the lightning model, which takes a Bias Detector Model architecture
#               and then defines processes required for training and testing of the model.
# Usage:        lightning_model = ModelWrapper(model_architecture=classifier, learning_rate=1e-3, loss=some_loss,
#                                dataset=some_dataset,
#                                dataset_distr=[int(0.5*len(some_dataset)), int(0.25*len(some_dataset)),
#                                int(0.25*len(some_dataset))],
#                                batch_size=batch_size)
# ------------

import numpy as np
import pytorch_lightning as pl
import torch

import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import statistics


class Max_Val(nn.Module):
    def __init__(self):
        super(Max_Val, self).__init__()

    def forward(self, x):
        zeros = torch.zeros(x.shape[0])
        zeros[[torch.argmax(x)]] = 1
        x = zeros
        return x


class ModelWrapper(pl.LightningModule):

    def __init__(self, model_architecture, learning_rate, loss, dataset=None, dataset_distr=None, test_dataset=None,
                 batch_size=1):
        super(ModelWrapper, self).__init__()
        self._model = model_architecture
        self.lr = learning_rate
        self.loss = loss
        if dataset is not None:
            self.dataset_split = random_split(dataset, dataset_distr)
        self.batch_size = batch_size
        self.test_dataset = test_dataset

        self.transform = Max_Val()

        # Warning: use below variable only when testing the model!
        self.test_accuracy = 0
        self.test_size = 0
        self.val_accuracy = 0
        self.val_size = 0

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch['model_weights']
        #y = batch['bias'].reshape(-1)
        # y = batch['bias']

        # Iterate over batch...
        new_list = self.dict_transform(x, batch['bias'].shape[0])
        losses = []
        for i in range(batch['bias'].shape[0]):
            y = batch['bias'][i]
            x = new_list[i]
            y_hat = self._model(x)
            # losses.append(self.loss(y_hat.float(), y.float()))
        #y_hat = self._model(x)
        #print(f'y: {y}\ny_hat: {y_hat}')

        return self.loss(y_hat.float(), y.float())
        # return statistics.mean(losses)

    def dict_transform(self, d, num_elements):
        res_list = []
        for i in range(num_elements):
            new_d = {}
            for el in d:
                new_d[el] = d[el][i].unsqueeze(0)
                # print(new_d[el].shape)
            res_list.append(new_d)
        return res_list

    def validation_step(self, batch, batch_idx):
        x = batch['model_weights']
        # Iterate over batch...
        new_list = self.dict_transform(x, batch['bias'].shape[0])
        for i in range(batch['bias'].shape[0]):
            y = batch['bias'][i]
            x = new_list[i]
            y_hat = self._model(x)
            accuracy = int(sum(y == self.transform.forward(y_hat))) == self._model.num_classes
            self.val_accuracy = self.val_accuracy + int(accuracy)
            self.val_size = self.val_size + 1
        return self.loss(y_hat.float(), y.float())

    def test_step(self, batch, batch_idx):
        x = batch['model_weights']

        # TODO: Implement batchwise training!!!
        new_list = self.dict_transform(x, batch['bias'].shape[0])
        losses = []
        for i in range(batch['bias'].shape[0]):
            y = batch['bias'][i]
            x = new_list[i]
            y_hat = self._model(x)
            loss = self.loss(y_hat.float(), y.float())
            y_hat = self.transform.forward(y_hat)
            accuracy = int(sum(y == y_hat)) == self._model.num_classes
            self.test_accuracy = self.test_accuracy + int(accuracy)
            self.test_size = self.test_size + 1
        return loss

    def on_test_end(self):
        print(f'correct: {self.test_accuracy}, total: {self.test_size}\nAccuracy: '
              f'{1.0*self.test_accuracy / self.test_size}')
        self._model.test_accuracy = 1.0*self.test_accuracy / self.test_size
        return {"Test Accuracy": 1.0*self.test_accuracy / self.test_size}

    def on_validation_end(self):
        print(f'\nValidation Accuracy: {1.0*self.val_accuracy / self.val_size}')
        self.val_accuracy = 0
        self.val_size = 0

    def train_dataloader(self):
        return DataLoader(self.dataset_split[0], batch_size=self.batch_size,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_split[1], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class MNIST_Classifier_Wrapper(pl.LightningModule):

    def __init__(self, model_architecture, learning_rate, loss, dataset=None, dataset_distr=None,
                 batch_size=1):
        super(MNIST_Classifier_Wrapper, self).__init__()
        self._model = model_architecture
        self.lr = learning_rate
        self.loss = loss
        if dataset is not None:
            self.dataset_split = random_split(dataset, dataset_distr)
        self.batch_size = batch_size

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        y_hat = self._model(x)
        return self.loss(y_hat.float(), y.float())

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        y_hat = self._model(x)
        return self.loss(y_hat.float(), y.float())

    def train_dataloader(self):
        return DataLoader(self.dataset_split[0], batch_size=self.batch_size,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_split[1], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_split[2], batch_size=self.batch_size)

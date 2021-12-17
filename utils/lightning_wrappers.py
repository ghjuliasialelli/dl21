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


class Max_Val(nn.Module):
    def __init__(self):
        super(Max_Val, self).__init__()

    def forward(self, x):
        zeros = torch.zeros(x.shape[0])
        zeros[[torch.argmax(x)]] = 1
        x = zeros
        return x


class ModelWrapper(pl.LightningModule):

    def __init__(self, model_architecture, learning_rate, loss, dataset=None, dataset_distr=None,
                 batch_size=1):
        super(ModelWrapper, self).__init__()
        self._model = model_architecture
        self.lr = learning_rate
        self.loss = loss
        if dataset is not None:
            self.dataset_split = random_split(dataset, dataset_distr)
        self.batch_size = batch_size

        self.transform = Max_Val()

        # Warning: use below variable only when testing the model!
        self.test_accuracy = 0
        self.test_size = 0

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch['model_weights']
        y = batch['bias'][0]
        y_hat = self._model(x)
        print(f'y: {y}\ny_hat: {y_hat}')
        return self.loss(y_hat.float(), y.float())

    def validation_step(self, batch, batch_idx):
        x = batch['model_weights']
        y = batch['bias'][0]
        # print(y.shape)
        y_hat = self._model(x)
        return self.loss(y_hat.float(), y.float())

    def test_step(self, batch, batch_idx):
        x = batch['model_weights']
        y = batch['bias'][0]
        y_hat = self._model(x)
        #y_hat = self.transform.forward(self._model(x))
        print(f'y: {y}\ny_hat: {y_hat}')
        loss = self.loss(y_hat.float(), y.float())
        y_hat = self.transform.forward(y_hat)

        # Change when changing number of classes to approximate
        # accuracy = int(sum(y == y_hat)) == 4
        accuracy = int(sum(y == y_hat)) == 2

        self.test_accuracy = self.test_accuracy + int(accuracy)
        self.test_size = self.test_size + 1
        return loss

    def on_test_end(self):
        print(f'correct: {self.test_accuracy}, total: {self.test_size}\nAccuracy: '
              f'{1.0*self.test_accuracy / self.test_size}')
        return {"Test Accuracy": 1.0*self.test_accuracy / self.test_size}

    def train_dataloader(self):
        return DataLoader(self.dataset_split[0], batch_size=self.batch_size,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_split[1], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_split[2], batch_size=self.batch_size)
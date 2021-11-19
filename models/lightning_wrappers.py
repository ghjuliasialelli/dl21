import pytorch_lightning as pl
import torch

from torch.nn import L1Loss


class ModelWrapper(pl.LightningModule):

    def __init__(self, model_architecture, learning_rate):
        self._model = model_architecture
        self.lr = learning_rate

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._model(x)
        loss = L1Loss()
        return loss(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

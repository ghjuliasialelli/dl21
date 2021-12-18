# ------------
# Author:       Philip Toma
# Description:  This file trains an arbitrary MNIST-classifier architecture.
# ------------


import pytorch_lightning as pl
import torch

from utils.model_operations import *
from utils.lightning_wrappers import *
from models.ifbid import *

import argparse

# define argument parser. The parser will be used when calling:
# python3 trainer.py --argument1 --argument2
parser = argparse.ArgumentParser(description='Run Model Training.')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs for model training.')
parser.add_argument('--debug', action='store_true',
                    help='set --debug flag for low computational impact.')
parser.add_argument('path', type=str, default='./',
                    help='path to the model data.')
args = parser.parse_args()

# Initialise model data
data = None

if args.debug:
    pass
    data_indices = list(range(0, len(data), 2))
    data = torch.utils.data.Subset(data, data_indices)

# Initialise the classifier model for training
batch_size = 1
classifier = None
loss = None

# Initialise pl model and trainer
lightning_model = MNIST_Classifier_Wrapper(model_architecture=classifier, learning_rate=1e-3, loss=loss, dataset=data,
                                           dataset_distr=[int(0.5*len(data)), int(0.25*len(data)), int(0.25*len(data))],
                                           batch_size=batch_size)

trainer = pl.Trainer(max_epochs=args.epochs, reload_dataloaders_every_n_epochs=2)
trainer.fit(lightning_model)
# Optional: Test the model
# trainer.test(lightning_model)

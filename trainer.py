# ------------
# Author:       Philip Toma
# Description:  This file implements the training pipeline of the IFBID Model.
# Usage:        python3 trainer.py [--debug] [--epochs 10] [path]
#               where the []-brackets mean an entry is optional, but should be used.
#               For more info:      python3 trainer.py --help
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
# data = ModelDataset('0.02', args.path)
data = PhilipsModelDataset('0.02', args.path, num_classes=4)
data.print_digit_classifier_info(0, True)

if args.debug:
    pass
    data_indices = list(range(0, len(data), 10))
    data = torch.utils.data.Subset(data, data_indices)

# get shapes for NN-initialisation:
m = data[0]['model_weights']

shapes = []
for layer in m:
    shapes.append(m[layer].shape)
print(shapes)

# Initialise the classifier model for training
batch_size = 12
# classifier = IFBID_Model(layer_shapes=shapes, batch_size=batch_size)
classifier = Dense_IFBID_Model(layer_shapes=shapes, use_dense=True, num_classes=4, batch_size=batch_size)
#classifier = Better_Dense(layer_shapes=shapes, use_dense=True, num_classes=4, batch_size=batch_size)
# classifier = Conv2D_IFBID_Model(layer_shapes=shapes, use_dense=True, num_classes=4, batch_size=batch_size)
#classifier = Max1D_IFBID_Model(layer_shapes=shapes, batch_size=batch_size)

#loss = torch.nn.L1Loss()
loss = torch.nn.BCELoss()
# loss = torch.nn.CrossEntropyLoss()

# Initialise the monitoring module.
# ...

# Initialise lightning checkpointing.

# Initialise pl model and trainer
lightning_model = ModelWrapper(model_architecture=classifier, learning_rate=1e-3, loss=loss, dataset=data,
                               dataset_distr=[int(0.5*len(data)), int(0.25*len(data)), int(0.25*len(data))],
                               batch_size=batch_size)
# print(lightning_model.forward(m))

trainer = pl.Trainer(max_epochs=args.epochs, reload_dataloaders_every_n_epochs=2)

# Train
trainer.fit(lightning_model) #, [train_loader, val_dataloader])


# Option: use the testing dataset (/test instead of /train) for further testing. Allows us to use more data for training.
# the bias doesn't matter for the below dataset.
# Test the model.
trainer.test(lightning_model)

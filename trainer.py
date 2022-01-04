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
from math import ceil
import csv

# define argument parser. The parser will be used when calling:
# python3 trainer.py --argument1 --argument2
parser = argparse.ArgumentParser(description='Run Model Training.')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs for model training.')
parser.add_argument('--debug', action='store_true',
                    help='set --debug flag for low computational impact.')
parser.add_argument('--stepsize', type=int, default=1,
                    help='control how much data is used for training.')
parser.add_argument('path', type=str, default='./',
                    help='path to the model data.')
args = parser.parse_args()

# Ensure Reproducability:
pl.seed_everything(2022, workers=True)

# Initialise model data. Set new_model=True to see how we train on the generalisation set
data = PhilipsModelDataset('0.02', os.path.join(args.path, 'train'), num_classes=4, standardize=False, new_model=False)
data_indices = list(range(0, len(data), args.stepsize))
data = torch.utils.data.Subset(data, data_indices)

if args.debug:
    data_indices = list(range(0, len(data), 100))
    data = torch.utils.data.Subset(data, data_indices)


# get MNIST-classifier shapes for NN-initialisation:
m = data[0]['model_weights']
shapes = []
for layer in m:
    shapes.append(m[layer].shape)
print(f'Layer-Shapes: {shapes}')

# Choose and Initialise the classifier model for training
batch_size = 2
# classifier = IFBID_Model(layer_shapes=shapes, batch_size=batch_size)
# classifier = Dense_IFBID_Model(layer_shapes=shapes, use_dense=True, num_classes=4, batch_size=batch_size)
classifier = Better_Dense(layer_shapes=shapes, use_dense=True, num_classes=4, batch_size=batch_size, new_model=False)
# classifier = Conv2D_IFBID_Model(layer_shapes=shapes, use_dense=True, num_classes=4, batch_size=batch_size,)

# Initialize training-loss
loss = torch.nn.BCELoss()

# Initialise lightning checkpointing.

# Initialise test_data.  Set new_model=True to see how we train on the generalisation set.
test_data = PhilipsModelDataset('0.02', os.path.join(args.path, 'test'), num_classes=4,
                                standardize=False, new_model=False)
data_indices = list(range(0, len(test_data), args.stepsize))
test_data = torch.utils.data.Subset(test_data, data_indices)

if args.debug:
    data_indices = list(range(0, len(test_data), 100))
    test_data = torch.utils.data.Subset(test_data, data_indices)


"""data, test_data = balance_datasets(train_data=data, test_data=test_data,
                                   split1=[int(0.7*len(data)), int(0.3*len(data))],
                                   split2=[int(0.7*len(test_data)), int(0.3*len(test_data))])"""

print("Sizes after Balancing")
print(f'Train-Data of length: {len(data)}, Test-Data of length {len(test_data)}')
print(f'{int(0.7*len(data))+int(0.3*len(data))}')

# Initialise pl model and trainer
lightning_model = ModelWrapper(model_architecture=classifier, learning_rate=1e-3, loss=loss, dataset=data,
                               dataset_distr=[int(0.7*len(data)), ceil(len(data) - 0.7*len(data))], test_dataset=test_data,
                               batch_size=batch_size)

trainer = pl.Trainer(max_epochs=args.epochs, deterministic=True, reload_dataloaders_every_n_epochs=2)

# Train the model.
trainer.fit(lightning_model)

# Test the model.
trainer.test(lightning_model)

# get and record accuracy obtained from test.
test_accuracy = lightning_model._model.test_accuracy

"""
with open(os.path.join(args.path, f'new_better_dense+dense-trainsize-{int(0.7*len(data))+int(0.7*len(test_data))}.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow([test_accuracy])

if not args.debug:
    os.makedirs(os.path.join(args.path, 'bias_classifiers'), exist_ok=True)
    torch.save(lightning_model._model.state_dict(),
               os.path.join(args.path, 'bias_classifiers',
                            f'new-trainsize-{int(0.7*len(data))+int(0.7*len(test_data))}')
               )"""

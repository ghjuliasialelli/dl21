# ------------
# Author:       Philip Toma
# Description:  This file implements the refinement pipeline of the Bias Classifier Models.
# Usage:        For info: python3 trainer.py --help
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

parser = argparse.ArgumentParser(description='Run Model Refinement.')
parser.add_argument('--model_number', type=int, default=10,
                    help='see README.md.')
parser.add_argument('--use_dense', action='store_true',
                    help='set flag for model to use dense layers.')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs for model training.')
parser.add_argument('--debug', action='store_true',
                    help='set --debug flag for low computational impact.')
parser.add_argument('initial_data_path', type=str, default='./',
                    help='path to the initial model data.')
parser.add_argument('path', type=str, default='./',
                    help='path to the model data.')
parser.add_argument('model_path', type=str, default='./',
                    help='path to the bias classifier.')
parser.add_argument('--test_on_old', action='store_true',
                    help='set flag to test refined Bias Classifier on authors MNIST-Classifiers.')
args = parser.parse_args()

# Ensure Reproducability:
pl.seed_everything(2022, workers=True)

# ------------------------------------------
# Load old model
# ------------------------------------------

data = PhilipsModelDataset('0.02', os.path.join(args.initial_data_path, 'test'), num_classes=4, standardize=False,
                           new_model=False)
old_len = len(data) # needed as quick fix for bug in dataset
m = data[0]['model_weights']
old_shapes = []
for layer in m:
    old_shapes.append(m[layer].shape)
print(f'Old Layer-Shapes: {old_shapes}')

"""TODO!!! Add model-number stuff!!"""

batch_size = 2

if args.model_number == 0:
    classifier = Conv2D_IFBID_Model(layer_shapes=old_shapes, use_dense=args.use_dense, num_classes=4,
                                    batch_size=batch_size, new_model=not args.test_on_old, two_layers=False)
    if args.use_dense:
        classifier.load_state_dict(torch.load(os.path.join(args.model_path, 'ifbid_for_refinement+dense-trainsize-700')))
    else:
        classifier.load_state_dict(torch.load(os.path.join(args.model_path, 'ifbid_for_refinement-trainsize-700')))
elif args.model_number == 1:
    classifier = Better_Dense(layer_shapes=old_shapes, use_dense=args.use_dense, num_classes=4, batch_size=batch_size,
                              new_model=not args.test_on_old)
    if args.use_dense:
        classifier.load_state_dict(torch.load(os.path.join(args.model_path, 'dense+dense-trainsize-700')))
    else:
        classifier.load_state_dict(torch.load(os.path.join(args.model_path, 'dense-trainsize-700')))


# ------------------------------------------
# Prepare Refinement
# ------------------------------------------

# Initialise model data
if not args.test_on_old:
    data = PhilipsModelDataset('0.02', os.path.join(args.path, 'train'), num_classes=4, standardize=False,
                               new_model=True)
else:
    data = PhilipsModelDataset('0.02', os.path.join(args.initial_data_path, 'test'),
                               num_classes=4, standardize=False, new_model=False)
    print([int(0.8*len(data)), ceil(0.2*len(data))])
    print(len(data))
    data, test_data = random_split(data, [int(0.8*len(data)), ceil(0.2*len(data))]) #[1, len(data)-1]) #

print(f'Refinement Dataset size: {len(data)}')

# get MNIST-classifier shapes for NN-initialisation:
m = data[0]['model_weights']
shapes = []
for layer in m:
    shapes.append(m[layer].shape)
print(f'Layer-Shapes: {shapes}')

# Initialize training-loss
loss = torch.nn.BCELoss()

# Initialise lightning checkpointing.

# Initialise test_data:
if not args.test_on_old:
    test_data = PhilipsModelDataset('0.02', os.path.join(args.path, 'test'), num_classes=4,
                                    standardize=False, new_model=True)

print(f'Refinement Test-Dataset size: {len(test_data)}')
if args.debug and args.test_on_old:
    data_indices = list(range(0, len(test_data), 10))
    test_data = torch.utils.data.Subset(test_data, data_indices)
    data_indices = list(range(0, len(data), 10))
    data = torch.utils.data.Subset(data, data_indices)


if not args.test_on_old:
    data, test_data = balance_datasets(train_data=data, test_data=test_data,
                                       split1=[int(0.8*len(data)), int(0.2*len(data))],
                                       split2=[int(0.8*len(test_data)), int(0.2*len(test_data))])

print("Sizes after Balancing")
print(f'Train-Data of length: {len(data)}, Test-Data of length {len(test_data)}')
print(f'{int(0.7*len(data))+int(0.3*len(data))}')

# Initialise pl model and trainer
lightning_model = ModelWrapper(model_architecture=classifier, learning_rate=1e-3, loss=loss, dataset=data,
                               dataset_distr=[int(0.7*len(data)), ceil(len(data) - 0.7*len(data))], test_dataset=test_data,
                               batch_size=batch_size)

trainer = pl.Trainer(max_epochs=args.epochs, deterministic=True) #, reload_dataloaders_every_n_epochs=2)

# ----------------------------------------------------------
# REFINEMENT OF THE PRETRAINED CLASSIFIER BELOW
# TRAIN-TEST-Slit with new models: 70%/30%.
# Models in new dataset were trained separately.
# ----------------------------------------------------------

trainer.fit(lightning_model)

# Test the model.
if args.test_on_old:
    lightning_model._model.last_index = len(old_shapes)-1

trainer.test(lightning_model)

test_accuracy = lightning_model._model.test_accuracy

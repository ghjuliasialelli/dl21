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
data = ModelDataset()

# Initialise the classifier model for training
classifier = IFBID_Model()

# Initialise the monitoring module.
# ...

# Initialise lightning checkpointing.

# Initialise pl model and trainer
lightning_model = ModelWrapper()
trainer = pl.Trainer()

# Train
# ...

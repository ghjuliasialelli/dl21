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
data = PhilipsModelDataset('0.02', args.path)
if args.debug:
    pass
    data_indices = list(range(0, len(data), 10))
    data = torch.utils.data.Subset(data, data_indices)

# train, val = PhilipsModelDataset('0.02', args.path), PhilipsModelDataset('0.03', args.path) # torch.utils.data.random_split(data, [0.8*len(data), 0.2*len(data)])
# train_loader = torch.utils.data.DataLoader(train)
# val_dataloader = torch.utils.data.DataLoader(val)


# get shapes for NN-initialisation:
m = data[0]['model_weights']

tensor_dim = np.sum([np.prod(m[layer].shape) for layer in m])
print(f'Size: {tensor_dim}')
#print(m)
shapes = []
for layer in m:
    shapes.append(m[layer].shape)
print(shapes)

# Initialise the classifier model for training
batch_size = 1
# classifier = IFBID_Model(layer_shapes=shapes, batch_size=batch_size)
# classifier = Dense_IFBID_Model(layer_shapes=shapes, batch_size=batch_size)
#classifier = Better_Dense(layer_shapes=shapes, batch_size=batch_size)
classifier = Conv2D_IFBID_Model(layer_shapes=shapes, batch_size=batch_size)
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

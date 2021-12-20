import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from utils.model_operations import LucasModelDataset
from models.ifbid import Conv2D_IFBID_Model2


NETWORK = 'conv2d_ifbid2'
DATASET = {
    'name': 'lucas_model_dataset',
    'use_weights': True,
    'use_biases': True,
    'only_conv': False
}
BATCH = 32
EPOCHS = 10
VAL_PER_EPOCHS = 5
SAVE_PER_EPOCHS = 5
SAVE_PATH = f"data/trained/{NETWORK}"


def plot_accuracy_and_loss(loss, acc):
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(np.arange(VAL_PER_EPOCHS, (len(acc) + 1) * VAL_PER_EPOCHS, VAL_PER_EPOCHS), acc)
    ax.set_title("Accuracy")
    ax.set_xlabel("epochs")
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(np.arange(1, len(loss) + 1), loss)
    ax.set_title("Loss")
    ax.set_xlabel("epochs")
    plt.tight_layout()
    plt.savefig(f'{SAVE_PATH}/loss.png')
    plt.close(fig)


def validation(dataloader, model):
    model.eval()

    with torch.no_grad():
        preds = []
        trues = []
        for data in dataloader:
            preds.append(np.argmax(model.forward(data['model_weights']).detach().cpu().numpy()[0]))
            trues.append(np.argmax(data['bias'].detach().cpu().numpy()[0]))
        preds = np.array(preds)
        trues = np.array(trues)

        # validation accuracy
        acc = np.count_nonzero(preds == trues) / len(preds)

    model.train()
    return acc


def train():
    # set random seed
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)

    # GPU support
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create dataloader
    if DATASET['name'] == 'lucas_model_dataset':
        train_dataset = LucasModelDataset(device, ['data/digitWdb_train.pkl.gz'], use_weights=DATASET['use_weights'],
                                          use_biases=DATASET['use_biases'], only_conv=DATASET['only_conv'])
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
        val_dataset = LucasModelDataset(device, ['data/digitWdb_test.pkl.gz'], use_weights=DATASET['use_weights'],
                                          use_biases=DATASET['use_biases'], only_conv=DATASET['only_conv'])
        val_dataloader = DataLoader(val_dataset, batch_size=1)
    else:
        raise ValueError("Dataset [{}] not recognized.".format(DATASET['name']))

    # create model
    layer_shapes = []
    for _, arr in train_dataset[0]['model_weights'].items():
        layer_shapes.append(arr.shape)
    if NETWORK == "conv2d_ifbid2":
        model = Conv2D_IFBID_Model2(layer_shapes, num_classes=4)
    else:
        raise ValueError("Network [{}] not recognized.".format(NETWORK))
    model.to(device)
    model.train()

    print('---------- Networks initialized -------------')
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(model)
    print('Total number of parameters: %d' % num_params)
    print('-----------------------------------------------')

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = torch.nn.BCELoss()

    loss_history = []
    acc_history = []

    # training
    for epoch in range(1, EPOCHS+1):
        train_start_time = time.time()
        train_running_loss = 0.0
        train_iter = 0
        for data in train_dataloader:
            optimizer.zero_grad()
            pred = model.forward(data['model_weights'])
            loss = loss_fcn(pred.reshape(-1), data['bias'].reshape(-1))
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            train_iter += 1

        if epoch % VAL_PER_EPOCHS == 0:
            accuracy = validation(val_dataloader, model)
            print(f"Epoch {epoch}: loss={(train_running_loss / train_iter):.5f}, acc={accuracy:.5f} time={(time.time() - train_start_time):.3f}s")
            acc_history.append(accuracy)
        else:
            print(f"Epoch {epoch}: loss={(train_running_loss / train_iter):.5f}, time={(time.time() - train_start_time):.3f}s")
        loss_history.append(train_running_loss / train_iter)

        # save model
        if epoch % SAVE_PER_EPOCHS == 0:
            model.save_network(SAVE_PATH, f"model-{epoch}")

    # save latest model
    model.save_network(SAVE_PATH, "model-latest")

    # validation on latest model
    accuracy = validation(val_dataloader, model)
    print(f"Final Accuracy: {accuracy:.5f}")

    plot_accuracy_and_loss(loss_history, acc_history)


import pandas as pd
from utils.model_operations import ModelDataset
import pyarrow as pa


def save_weights_as_pickle():
    # load training data
    train = pd.DataFrame()
    for b in ['0.02', '0.03', '0.04', '0.05']:
        model_data = ModelDataset(bias=b, data_directory='data/DigitWdb/train')
        for model_idx in range(len(model_data)):
            print(f"Train ({b}) Model {model_idx+1}/{len(model_data)}")
            model = model_data[model_idx]
            layers = []
            for layer in model.layers:
                if len(layer.get_weights()) > 0:
                    layers.append({
                        'name': layer.__class__.__name__,
                        'weights': layer.get_weights()[0],
                        'bias': layer.get_weights()[1]
                    })
            train = train.append({'model': model_idx, 'layers': layers, 'bias': b}, ignore_index=True)

    train.to_pickle("data/digitWdb_train.pkl.gz", compression='gzip')

    # load testing data
    test = pd.DataFrame()
    for b in ['0.02', '0.03', '0.04', '0.05']:
        model_data = ModelDataset(bias=b, data_directory='data/DigitWdb/test')
        for model_idx in range(len(model_data)):
            print(f"Test ({b}) Model {model_idx+1}/{len(model_data)}")
            model = model_data[model_idx]
            layers = []
            for layer in model.layers:
                if len(layer.get_weights()) > 0:
                    layers.append({
                        'name': layer.__class__.__name__,
                        'weights': layer.get_weights()[0],
                        'bias': layer.get_weights()[1]
                    })

            test = test.append({'model': model_idx, 'layers': layers, 'bias': b}, ignore_index=True)

    test.to_pickle("data/digitWdb_test.pkl.gz", compression='gzip')


if __name__ == '__main__':
    train()

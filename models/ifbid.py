# ------------
# Author:       Philip Toma
# Description:  This file implements the IFBID Model: a models for inference-free bias detection learning
#               The paper can be found at https://arxiv.org/abs/2109.04374
# ------------
import torch
from torch import nn

from collections import OrderedDict


class Print(nn.Module):
    def __init__(self, i):
        super(Print, self).__init__()
        self.i = i

    def forward(self, x):
        print(f'i: {self.i}.. Shape: {x.shape}')
        return x


class IFBID_Model(nn.Module):
    """
    Input to the model is
    """

    def __init__(self):
        super(IFBID_Model, self).__init__()
        self.flatten = nn.Flatten()
        """self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )"""

    def forward(self, model):
        """
        Input to the ifbid-model is a model.
        Shape of weights should be something like: (Layers, Weight-shape at Layer i)
        """
        # for each layer:
        #   convolutional block
        #   somehow concat the ouputs
        # return the concated output.
        feature_list = []
        for layer in model.layers:
            # [0] are weights, [1] are biases.
            weights = layer.get_weights()
            if layer.__class__.__name__ == 'Conv2D':# or layer.__class__.__name__ == 'Dense':
                ws = torch.permute(torch.from_numpy(weights[0]), (3, 2, 0, 1))
                y = self.simple_conv_block(ws)
            else:
                # what to do?!
                # y = self.simple_conv_block(torch.FloatTensor(weights))
                continue
            feature_list.append(y)
        features = torch.cat(feature_list)
        # No loss function defined. I will use l1 (MAE) in the training process for now.
        # dense layer at the end.
        features = features.squeeze()
        linear = nn.Linear(in_features=(features.shape[0]), out_features=1)
        return linear(features)

    def simple_conv_block(self, x, s=10):
        # s (in Fig. 2 it's called m) defines the number of features that we want to return
        # x = self.flatten(x)
        x = x.flatten()
        linear = nn.Linear(in_features=(x.shape[0]), out_features=s)
        return linear(x)

    def debugging_cb(self, x, c):
        d = x.shape[-1]
        s = x.shape[0]

        layer_1 = nn.Conv2d(in_channels=x.shape[1], out_channels=c, kernel_size=(1, 1))
        layer_2 = nn.MaxPool2d(kernel_size=(d, d))
        extra_layer = nn.Flatten(-2)
        layer_3 = nn.Conv1d(in_channels=c, out_channels=c, kernel_size=(1,))
        layer_4 = nn.MaxPool1d(c)
        flatten = nn.Flatten()

        pipeline = nn.Sequential(
            OrderedDict([
                ('Layer1', layer_1),
                ('Layer2', layer_2),
                ('Flatten_1', extra_layer),
                ('Layer3', layer_3),
                ('Flatten_2', nn.Flatten()),
                ('Layer4', layer_4)
            ])
        )
        return pipeline

    def convolutional_block(self, x, c, k=None):
        """Convolutions are followed by a relu activation function,
        and there is always 0.1 dropout afterwards"""
        d = x.shape[-1]
        print(d)
        s = x.shape[0]
        if len(x.shape) > 2:
            # in convolutional layer (cl), either dxd or 1x1 kernel size, as seen on pg. 5
            layer_1 = nn.Conv2d(in_channels=x.shape[1], out_channels=c, kernel_size=(1, 1))
            # x = layer_1(x)
            if k is not None:
                layer_2 = nn.MaxPool3d(kernel_size=(d, d, k))
            else:
                layer_2 = nn.MaxPool2d(kernel_size=(d, d))
            # or:
        else:
            return None
            # layer_1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=(d, d))
            # layer_2 = None
        # MaxPooling2D
        # Conv1D
        # MaxPooling1D
        # Flatten
        if k is not None:
            return nn.Sequential(
                OrderedDict([
                    ('Layer1', layer_1),
                    ('Layer2', layer_2),
                    ('Layer3', nn.Flatten())
                ])
            )
        else:
            # layer 3: how to work with in/out channels?
            extra_layer = nn.Flatten(-2)
            layer_3 = nn.Conv1d(in_channels=c, out_channels=c, kernel_size=(1,))
            layer_4 = nn.MaxPool1d(c)
            """return nn.Sequential(
                OrderedDict([
                    ('Layer1', layer_1),
                    ('Layer2', layer_2),
                    ('Flatten', extra_layer),
                    ('Layer3', layer_3),
                    ('Flatten2', nn.Flatten()),
                    ('Debug-Info', Print()),
                    ('Layer4', layer_4)
                ])
            )"""
            pipeline = nn.Sequential(
                OrderedDict([
                    ('Layer1', layer_1),
                    ('Layer2', layer_2),
                    ('Flatten_1', extra_layer),
                    ('Layer3', layer_3),
                    ('Flatten_2', nn.Flatten()),
                    ('Layer4', layer_4)
                ])
            )
            return pipeline

    def __str__(self):
        pass

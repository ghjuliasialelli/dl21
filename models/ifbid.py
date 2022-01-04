# ------------
# Author:       Philip Toma
# Description:  This file implements the IFBID Model: a models for inference-free bias detection learning
#               The paper can be found at https://arxiv.org/abs/2109.04374
# ------------
import os.path

import numpy as np
import torch
from torch import nn

from collections import OrderedDict


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(f'Shape: {x.shape}')
        return x


class Max_Val(nn.Module):
    def __init__(self):
        super(Max_Val, self).__init__()

    def forward(self, x):
        zeros = torch.zeros(4)
        zeros[[torch.argmax(x)]] = 1
        x = zeros
        return x


"""
TODO: Need to fix the way we give the input. Can't just give a tensor. The way we did it until now
was fine in my opinion. But datasplit doesnt work.
"""


class IFBID_Model(nn.Module):
    """
    Input to the model is
    """

    def __init__(self, layer_shapes, use_dense_layers=False, num_classes=2, batch_size=1):
        super(IFBID_Model, self).__init__()
        self.layer_shapes = layer_shapes
        # print(f'Shape: {layer_shapes[0]}')

        # TODO: CHANGE THE LAYOUT of layers. Make an init
        print(f'Calculated Shape: {np.prod(layer_shapes[0])}')
        print([np.prod(layer_shapes[i]) for i in range(len(layer_shapes))])
        print(np.sum([np.prod(layer_shapes[i]) for i in range(len(layer_shapes))]))
        self.layers = None
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.use_dense_layers = use_dense_layers
        """self.layers = nn.Sequential(
            #nn.Flatten(start_dim=0, end_dim=-1),
            nn.Linear(in_features=(int(batch_size*np.sum([np.prod(layer_shapes[i])
                                                          for i in range(len(layer_shapes))]))),
                      out_features=4),
            #nn.Sigmoid()
            # nn.ReLU()
            torch.nn.Softmax()
        )"""

    def init_layers(self, type):
        if type == "SIMPLE":
            pass
        elif type == "2D":
            pass
        elif type == "3D":
            pass
        return nn.Sequential()

    def forward(self, model):
        """
        Input to the ifbid-model is a model.
        Shape of weights should be something like: (Layers, Weight-shape at Layer i)
        """

        # Build a flattened tensor of all layers.
        layer_tensor = []
        for layer in model:
            layer_tensor.append(model[layer].flatten())
        layer_tensor = torch.cat(layer_tensor)
        return self.layers(layer_tensor.flatten())

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
        and there is always 0.1 dropout afterwards
        @:param     x is the """
        # ---
        # Usage:
        #   c sets channels of convolutions. -> Number of convolutional filters that our output by the conv.
        #   !!! if using k, then set k = c.
        #   WHEN DEBUGGING: USE ('Debug{i}', Print(i)) in the return OrderedDict to print dimension of x at layer i.
        # ---

        d = x[-1]
        s = x[0]
        if len(x) > 2:
            # in convolutional layer (cl), either dxd or 1x1 kernel size, as seen on pg. 5
            layer_1 = nn.Conv2d(in_channels=x[1], out_channels=c, kernel_size=(1, 1))
            # x = layer_1(x)
            if k is not None:
                layer_2 = nn.MaxPool3d(kernel_size=(k, d, d))
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
                    ('Layer3', nn.Flatten()),
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


class Dense_IFBID_Model(IFBID_Model):

    def __init__(self, layer_shapes, use_dense, num_classes, batch_size=1):
        super(Dense_IFBID_Model, self).__init__(layer_shapes, use_dense_layers=use_dense, num_classes=num_classes,
                                                batch_size=batch_size)
        self.layers = nn.Sequential(
            # nn.Flatten(start_dim=0, end_dim=-1),
            nn.Linear(in_features=(int(np.sum([np.prod(layer_shapes[i]) for i in range(len(layer_shapes))]))),
                      out_features=self.num_classes),
            # nn.Sigmoid()
            # nn.ReLU()
            torch.nn.Softmax()
        )

    def forward(self, model):
        """
        Input to the ifbid-model is a model.
        Shape of weights should be something like: (Layers, Weight-shape at Layer i)
        """

        # Build a flattened tensor of all layers.
        layer_tensor = []
        for layer in model:
            layer_tensor.append(model[layer].flatten())
        layer_tensor = torch.cat(layer_tensor)

        return self.layers(layer_tensor.flatten())


class Better_Dense(IFBID_Model):
    """
    Dense Convolutional Blocks, as implemented in Paper.
    Dropout probably only starts working once the batch size is large enough to stop
    overfitting.
    """

    def __init__(self, layer_shapes, use_dense, num_classes=2, batch_size=1, new_model=False):
        super(Better_Dense, self).__init__(layer_shapes, use_dense, num_classes, batch_size)

        if new_model:
            self.last_index = len(layer_shapes)
        else:
            self.last_index = len(layer_shapes)-1

        self.blocks = []
        self.block_0 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(in_features=(int(np.prod(layer_shapes[0]))),
                      out_features=300),
            # nn.Sigmoid()
            nn.ReLU(),
            nn.Dropout(p=0.1)
            # torch.nn.Softmax()
        )
        self.block_1 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(in_features=(int(np.prod(layer_shapes[1]))),
                      out_features=300),
            # nn.Sigmoid()
            nn.ReLU(),
            nn.Dropout(p=0.1)
            # torch.nn.Softmax()
        )
        self.block_2 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(in_features=(int(np.prod(layer_shapes[2]))),
                      out_features=300),
            # nn.Sigmoid()
            nn.ReLU(),
            nn.Dropout(p=0.1)
            # torch.nn.Softmax()
        )
        m2 = 0
        if use_dense:
            m2 = 100
            self.block_3 = nn.Sequential(
                nn.Linear(in_features=(int(np.prod(layer_shapes[-1]))),
                          out_features=m2),
                # nn.Sigmoid()
                nn.ReLU(),
                nn.Dropout(p=0.1)
                # torch.nn.Softmax()
            )
        self.final_dense = nn.Sequential(
            # nn.Linear(in_features=900, out_features=4),
            nn.Linear(in_features=900+m2, out_features=self.num_classes),
            # nn.ReLU()
            torch.nn.Softmax()
        )

    def forward(self, model):
        """
        Input to the ifbid-model is a model.
        Shape of weights should be something like: (Layers, Weight-shape at Layer i)
        """
        # Build a flattened tensor of all layers.
        layer_tensor = []
        # for layer in model:
        layer_tensor.append(self.block_0(model['layer_0'].flatten()))
        layer_tensor.append(self.block_1(model['layer_1'].flatten()))
        layer_tensor.append(self.block_2(model['layer_2'].flatten()))
        if self.use_dense_layers:
            #layer_tensor.append(self.block_3(model['layer_3'].flatten()))
            layer_tensor.append(self.block_3(model[f'layer_{self.last_index}'].flatten()))
        layer_tensor = torch.cat(layer_tensor)
        return self.final_dense(layer_tensor)


# Helper for Conv2D Model:
class Reshaper(nn.Module):

    def __init__(self, dim):
        super(Reshaper, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.reshape(self.dim)
        return x.unsqueeze(0)
        # return x


class Conv2D_IFBID_Model(IFBID_Model):

    def __init__(self, layer_shapes, use_dense, num_classes, batch_size=1, new_model=False, two_layers=True):
        super(Conv2D_IFBID_Model, self).__init__(layer_shapes, use_dense_layers=use_dense, num_classes=num_classes,
                                                 batch_size=batch_size)
        if new_model:
            self.last_index = len(layer_shapes)
        else:
            self.last_index = len(layer_shapes)-1
        self.two_layers = two_layers
        """d = x[-1]
        s = x[0]
        if len(x) > 2:
            # in convolutional layer (cl), either dxd or 1x1 kernel size, as seen on pg. 5
            layer_1 = nn.Conv2d(in_channels=x[1], out_channels=c, kernel_size=(1, 1))
            else:
                layer_2 = nn.MaxPool2d(kernel_size=(d, d))"""
        m = 100
        mp = 0
        if use_dense:
            m2 = 10
        else:
            m2 = 0
        self.block_0 = self.build_block(0, layer_shapes[0], c=100, d=3, m=m)
        self.block_1 = self.build_block(1, layer_shapes[1], c=100, d=3, m=m)
        self.block_2 = self.build_block(2, layer_shapes[2], c=100, d=3, m=m)

        # The info below is not used in the baseline!
        if use_dense:
            if self.two_layers:
                self.block_3 = self.build_d_block(3, layer_shapes[3], m=m2)
                mp = 2*m2
            else:
                mp = m2
            self.block_4 = self.build_d_block(4, layer_shapes[-1], m=m2)

        self.final_dense = nn.Sequential(
            nn.Linear(in_features=3*m + mp, out_features=self.num_classes),
            # in features should be 3*actual_m
            # nn.Linear(in_features=12, out_features=4),
            # nn.ReLU()
            torch.nn.Softmax()
        )

    def forward(self, model):
        """
        Input to the ifbid-model is a model.
        Shape of weights should be something like: (Layers, Weight-shape at Layer i)
        """
        # Build a flattened tensor of all layers.
        output_tensor = []
        # for layer in model:
        shape = model['layer_0'].shape
        output_tensor.append(self.block_0(model['layer_0'].reshape((shape[1], shape[2], shape[3], shape[4]))))
        shape = model['layer_1'].shape
        output_tensor.append(self.block_1(model['layer_1'].reshape((shape[1], shape[2], shape[3], shape[4]))))
        shape = model['layer_2'].shape
        output_tensor.append(self.block_2(model['layer_2'].reshape((shape[1], shape[2], shape[3], shape[4]))))

        # BELOW NOT USED FOR BASELINE!
        if self.use_dense_layers:
            if self.two_layers:
                output_tensor.append(self.block_3(model['layer_3']))
            output_tensor.append(self.block_4(model[f'layer_{self.last_index}']))

        for i in range(len(output_tensor)):
            output_tensor[i] = output_tensor[i].squeeze()

        output_tensor = torch.cat(output_tensor)
        #print(output_tensor.shape)
        return self.final_dense(output_tensor)

    def build_d_block(self, i, layer_shapes, m):
        return nn.Sequential(
            Reshaper(-1),
            nn.Linear(in_features=layer_shapes[0]*layer_shapes[1], out_features=m),
            nn.ReLU()
        )

    def build_block(self, i, layer_shapes, c, d, m):
        layer_1 = nn.Conv2d(in_channels=layer_shapes[1], out_channels=c, kernel_size=(1, 1))
        layer_2 = nn.MaxPool2d(kernel_size=(d, d))
        layer_3 = nn.Conv1d(in_channels=c, out_channels=m, kernel_size=(1, 1))
        layer_4 = nn.MaxPool1d(layer_shapes[0])

        #print(layer_shapes)
        pipeline = nn.Sequential(
            OrderedDict([
                ('Layer1', layer_1),
                ('Activation_1', nn.ReLU()),
                ('Layer2', layer_2),
                ('Layer3', layer_3),
                # ('Printer', Print()),
                ('Activation_2', nn.ReLU()),
                ('Flatten_2', nn.Flatten()),
                ('Reshape', Reshaper(-1)),
                ('Layer4', layer_4),
            ])
        )
        return pipeline


class Reshaper2(nn.Module):

    def __init__(self, reshape_dim=None, perm_dim=None):
        super(Reshaper2, self).__init__()
        self.reshape_dim = reshape_dim
        self.perm_dim = perm_dim

    def forward(self, x):
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        if self.reshape_dim is not None:
            x = x.reshape(self.reshape_dim)
        if self.perm_dim is not None:
            #x = torch.permute(x, self.perm_dim)
            x = x.permute(self.perm_dim)
        return x


class Conv2D_IFBID_Model2(nn.Module):

    def __init__(self, layer_shapes, num_classes):
        super(Conv2D_IFBID_Model2, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(layer_shapes)

        n_count = 0

        m = 100
        m2 = 10
        n = 100
        n2 = 10
        self.block_0 = self.build_block(0, layer_shapes[0], n=n if np.prod(layer_shapes[0][1:]) > 1 else n2,
                                        m=m if np.prod(layer_shapes[0][1:]) > 1 else m2)
        n_count += m if np.prod(layer_shapes[0][1:]) > 1 else m2
        self.block_1 = self.build_block(1, layer_shapes[1], n=n if np.prod(layer_shapes[1][1:]) > 1 else n2,
                                        m=m if np.prod(layer_shapes[1][1:]) > 1 else m2)
        n_count += m if np.prod(layer_shapes[1][1:]) > 1 else m2
        self.block_2 = self.build_block(2, layer_shapes[2], n=n if np.prod(layer_shapes[2][1:]) > 1 else n2,
                                        m=m if np.prod(layer_shapes[2][1:]) > 1 else m2)
        n_count += m if np.prod(layer_shapes[2][1:]) > 1 else m2
        if self.num_layers > 3:
            self.block_3 = self.build_block(3, layer_shapes[3], n=n if np.prod(layer_shapes[3][1:]) > 1 else n2,
                                            m=m if np.prod(layer_shapes[3][1:]) > 1 else m2)
            n_count += m if np.prod(layer_shapes[3][1:]) > 1 else m2
            self.block_4 = self.build_block(4, layer_shapes[4], n=n if np.prod(layer_shapes[4][1:]) > 1 else n2,
                                            m=m if np.prod(layer_shapes[4][1:]) > 1 else m2)
            n_count += m if np.prod(layer_shapes[4][1:]) > 1 else m2
        if self.num_layers > 5:
            self.block_5 = self.build_block(5, layer_shapes[5], n=n if np.prod(layer_shapes[5][1:]) > 1 else n2,
                                            m=m if np.prod(layer_shapes[5][1:]) > 1 else m2)
            n_count += m if np.prod(layer_shapes[5][1:]) > 1 else m2
        if self.num_layers > 6:
            self.block_6 = self.build_block(6, layer_shapes[6], n=n if np.prod(layer_shapes[6][1:]) > 1 else n2,
                                            m=m if np.prod(layer_shapes[6][1:]) > 1 else m2)
            n_count += m if np.prod(layer_shapes[6][1:]) > 1 else m2
            self.block_7 = self.build_block(7, layer_shapes[7], n=n if np.prod(layer_shapes[7][1:]) > 1 else n2,
                                            m=m if np.prod(layer_shapes[7][1:]) > 1 else m2)
            n_count += m if np.prod(layer_shapes[7][1:]) > 1 else m2
            self.block_8 = self.build_block(8, layer_shapes[8], n=n if np.prod(layer_shapes[8][1:]) > 1 else n2,
                                            m=m if np.prod(layer_shapes[8][1:]) > 1 else m2)
            n_count += m if np.prod(layer_shapes[8][1:]) > 1 else m2
            self.block_9 = self.build_block(9, layer_shapes[9], n=n if np.prod(layer_shapes[9][1:]) > 1 else n2,
                                            m=m if np.prod(layer_shapes[9][1:]) > 1 else m2)
            n_count += m if np.prod(layer_shapes[9][1:]) > 1 else m2

        self.final_dense = nn.Sequential(
            nn.Linear(in_features=n_count, out_features=self.num_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, model):
        """
        Input to the ifbid-model is a model.
        Shape of weights should be something like: (Layers, Weight-shape at Layer i)
        """

        # Build a flattened tensor of all layers.
        output_tensor = []

        output_tensor.append(self.block_0(model['layer_0']))
        output_tensor.append(self.block_1(model['layer_1']))
        output_tensor.append(self.block_2(model['layer_2']))
        if self.num_layers > 3:
            output_tensor.append(self.block_3(model['layer_3']))
            output_tensor.append(self.block_4(model['layer_4']))
        if self.num_layers > 5:
            output_tensor.append(self.block_5(model['layer_5']))
        if self.num_layers > 6:
            output_tensor.append(self.block_6(model['layer_6']))
            output_tensor.append(self.block_7(model['layer_7']))
            output_tensor.append(self.block_8(model['layer_8']))
            output_tensor.append(self.block_9(model['layer_9']))

        output_tensor = torch.cat(output_tensor, dim=1)
        # print(output_tensor.shape)
        return self.final_dense(output_tensor)

    def build_block(self, i, layer_shapes, n, m):
        layers = []
        layer_1 = nn.Conv2d(in_channels=layer_shapes[1], out_channels=n, kernel_size=(1, 1))
        layers.extend([('Reshape1', Reshaper2(reshape_dim=(-1, *layer_shapes[1:]))),
                       ('Layer1', layer_1), ('Activation_1', nn.ReLU())])
        if layer_shapes[2] > 1:
            layer_2 = nn.MaxPool2d(kernel_size=(layer_shapes[2], layer_shapes[3]))
            layers.append(('Layer2', layer_2))
        layers.append(('Flatten', nn.Flatten(start_dim=2)))
        layer_3 = nn.Conv1d(in_channels=n, out_channels=m, kernel_size=1)
        layers.extend([('Layer3', layer_3), ('Activation_2', nn.ReLU())])
        layer_4 = nn.MaxPool1d(layer_shapes[0])
        layers.extend([('Reshape2', Reshaper2(reshape_dim=(-1, layer_shapes[0], m), perm_dim=(0, 2, 1))),
                       ('Layer4', layer_4), ('Reshape3', Reshaper2())])

        #print(layer_shapes)
        pipeline = nn.Sequential(OrderedDict(layers))
        return pipeline

    def save_network(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        save_filename = f'{filename}.pth'
        torch.save(self.state_dict(), os.path.join(path, save_filename))



class Max1D_IFBID_Model(IFBID_Model):

    def __init__(self, layer_shapes, num_classes, batch_size=1):
        super(Max1D_IFBID_Model, self).__init__(layer_shapes, num_classes, batch_size=1)

        """d = x[-1]
        s = x[0]
        if len(x) > 2:
            # in convolutional layer (cl), either dxd or 1x1 kernel size, as seen on pg. 5
            layer_1 = nn.Conv2d(in_channels=x[1], out_channels=c, kernel_size=(1, 1))
            else:
                layer_2 = nn.MaxPool2d(kernel_size=(d, d))"""
        m = 100
        self.block_0 = self.build_block(0, layer_shapes[0], c=100, d=3, m=m, k=5)
        self.block_1 = self.build_block(1, layer_shapes[1], c=100, d=3, m=m, k=5)
        self.block_2 = self.build_block(2, layer_shapes[2], c=100, d=3, m=m, k=5)
        self.final_dense = nn.Sequential(
            nn.Linear(in_features=3 * m, out_features=self.num_classes * batch_size),
            # in features should be 3*actual_m
            # nn.Linear(in_features=12, out_features=4),
            # nn.ReLU()
            torch.nn.Softmax()
        )

    def forward(self, model):
        """
        Input to the ifbid-model is a model.
        Shape of weights should be something like: (Layers, Weight-shape at Layer i)
        """

        # Build a flattened tensor of all layers.
        output_tensor = []
        # for layer in model:
        output_tensor.append(self.block_0(model['layer_0'].squeeze()))
        output_tensor.append(self.block_1(model['layer_1'].squeeze()))
        output_tensor.append(self.block_2(model['layer_2'].squeeze()))
        for i in range(len(output_tensor)):
            output_tensor[i] = output_tensor[i].squeeze()
        """for el in output_tensor:
            print(el.shape)"""
        output_tensor = torch.cat(output_tensor)
        return self.final_dense(output_tensor)

    def build_block(self, i, layer_shapes, c, d, m, k):
        layer_1 = nn.Conv2d(in_channels=layer_shapes[1], out_channels=c, kernel_size=(1, 1))
        layer_2 = nn.MaxPool3d(kernel_size=(c, d, d))

        print(layer_shapes)
        pipeline = nn.Sequential(
            OrderedDict([
                ('Pre-Printer0', Print()),
                ('Layer1', layer_1),
                ('Activation_1', nn.ReLU()),
                # ('Activation_1', nn.Softmax()),
                # ('Reshape', Reshaper(-1)),
                ('Printer0', Print()),
                ('Layer2', layer_2),
                ('Flatten', nn.Flatten()),
                ('Printer', Print()),
                ('Activation_2', nn.ReLU()),
                # ('Activation_2', nn.Softmax()),
            ])
        )
        return pipeline

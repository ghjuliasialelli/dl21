import math
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd
from torch.nn.init import kaiming_uniform_
import time


# DATASETS

def load_network_weights_dataset(options):
    from utils.model_operations import LucasModelDataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return LucasModelDataset(device, ['data/digitWdb_train.pkl.gz', 'data/digitWdb_test.pkl.gz'],
                             use_weights=options['use_weights'], use_biases=options['use_weights'], only_conv=False)


def load_unbiased_color_mnist_dataset(train=False):
    from utils.image_operations import LucasDigitData
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not train:
        return LucasDigitData(device, [
            'data/colored_mnist/mnist_10color_jitter_var_0.050.npy'
        ], train=False, test=True)
    else:
        return LucasDigitData(device, [
            'data/colored_mnist/mnist_10color_jitter_var_0.045.npy'
        ], train=False, test=True)


# NETWORKS

def load_network(options, layer_shapes):
    # create network
    if options['network'] == 'conv2d_ifbid':
        from models.ifbid import Conv2D_IFBID_Model2
        net = Conv2D_IFBID_Model2(layer_shapes, 4)
    elif options['network'] == 'rnn':
        raise NotImplementedError()
        # TODO load recurrent neural network
    else:
        raise ValueError("Network [{}] not recognized.".format(options['network']))

    # load network to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # load pretrained network weights
    state_dict = torch.load(options['pretrained_network_weights_path'], map_location=device)
    net.load_state_dict(state_dict, strict=True)

    return net


def get_digit_classifier(weights):
    from models.MNIST_classifiers import MNISTClassifier
    classifier = MNISTClassifier()
    classifier.set_weights(weights)

    # load network to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    return classifier


# BIAS METRICS

def accuracy_on_unbiased_mnist(net):
    # load unbiased colored MNIST dataset
    mnist_dataset = load_unbiased_color_mnist_dataset()
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=1)

    net.eval()

    with torch.no_grad():
        trues = []
        preds = []
        for X, y in mnist_dataloader:
            preds.append(np.argmax(net(X).detach().cpu().numpy()[0]))
            trues.append(y.detach().cpu().numpy()[0, 0])
        preds = np.array(preds)
        trues = np.array(trues)

        accuracy = np.count_nonzero(preds == trues) / len(preds)

    return accuracy


def score_ratio(net):
    meanColors = np.array([
        [220, 20, 60],
        [0, 128, 128],
        [253, 233, 16],
        [0, 149, 182],
        [237, 145, 33],
        [145, 30, 188],
        [70, 240, 240],
        [250, 197, 187],
        [210, 245, 60],
        [128, 0, 0]
    ])

    # load unbiased colored MNIST dataset
    mnist_dataset = load_unbiased_color_mnist_dataset()
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=1)

    net.eval()

    with torch.no_grad():
        bias_preds = []
        preds = []
        count = [0]*10
        for X, y in mnist_dataloader:
            true = y.detach().cpu().numpy()[0, 0]
            if count[true] < 20:
                pred = net(change_mnist_color(meanColors[true], X)).detach().cpu().numpy()[0]
                if pred[true] >= 0.75:
                    bias_preds.append(pred[true])
                    for r in range(15, 315, 75):
                        for g in range(15, 315, 75):
                            for b in range(15, 315, 75):
                                if 50 <= r+g+b <= 715:
                                    pred = net(change_mnist_color([r, g, b], X)).detach().cpu().numpy()[0]
                                    preds.append(pred[true])
                    count[true] += 1

        ratio = np.mean(preds) / np.mean(bias_preds)

    return ratio


def change_mnist_color(color, image):
    color_map = torch.logical_or(torch.logical_or(image[:, 0] > 0, image[:, 1] > 0), image[:, 2] > 0)
    new_color = image.clone()
    for i in range(3):
        new_color[:, i][color_map] = color[i]
    return new_color


# TODO


# LOCALIZATION/MITIGATION METHODS

def fast_gradient_sign_method(net, data, epsilon=2e-2, steps=1, layers=None, neurons=None):
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fcn = torch.nn.CrossEntropyLoss()

    # copy data data
    new_weights = {}
    for k, v in data['model_weights'].items():
        new_weights[k] = v.clone().requires_grad_()
    new_data = {'model_weights': new_weights, 'bias': data['bias'].clone()}

    # prediction before
    with torch.no_grad():
        prediction_before = net(data['model_weights']).detach().cpu().numpy()[0]

    for param in net.parameters():
        param.requires_grad = False

    # FGSM steps
    for i in range(steps):
        pred = net(new_data['model_weights'])
        loss = loss_fcn(pred, torch.tensor([3], device=device).long())
        loss.backward()

        for layer in new_data['model_weights']:
            if layers is None or layer in layers:
                if neurons is None:
                    sign_grad = torch.sign(new_data['model_weights'][layer].grad.to(device))
                else:
                    sign_grad = torch.sign(new_data['model_weights'][layer].grad.to(device)) * torch.from_numpy(neurons[layer]).to(device)
                with torch.no_grad():
                    new_data['model_weights'][layer] -= epsilon * sign_grad
            new_data['model_weights'][layer].grad.data.zero_()

    # no grad on weights
    for layer in new_data['model_weights']:
        new_data['model_weights'][layer].requires_grad_(False)

    # create mnist classifier
    digit_net = get_digit_classifier(new_data['model_weights'])

    # prediction after
    with torch.no_grad():
        prediction_after = net(new_data['model_weights']).detach().cpu().numpy()[0]

    return digit_net, new_data, prediction_before, prediction_after


def gradient_method(net, data, epsilon=2e-2, steps=1, layers=None, neurons=None):
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fcn = torch.nn.CrossEntropyLoss()

    # copy data data
    new_weights = {}
    for k, v in data['model_weights'].items():
        new_weights[k] = v.clone().requires_grad_()
    new_data = {'model_weights': new_weights, 'bias': data['bias'].clone()}

    # prediction before
    with torch.no_grad():
        prediction_before = net(data['model_weights']).detach().cpu().numpy()[0]

    for param in net.parameters():
        param.requires_grad = False

    # FGSM steps
    for i in range(steps):
        pred = net(new_data['model_weights'])
        loss = loss_fcn(pred, torch.tensor([3], device=device).long())
        loss.backward()

        for layer in new_data['model_weights']:
            if layers is None or layer in layers:
                if neurons is None:
                    grad = new_data['model_weights'][layer].grad.clone().to(device)
                else:
                    grad = new_data['model_weights'][layer].grad.clone().to(device) * torch.from_numpy(neurons[layer]).to(device)
                with torch.no_grad():
                    new_data['model_weights'][layer] -= epsilon * grad / torch.max(torch.abs(grad))
            new_data['model_weights'][layer].grad.data.zero_()

    # no grad on weights
    for layer in new_data['model_weights']:
        new_data['model_weights'][layer].requires_grad_(False)

    # create mnist classifier
    digit_net = get_digit_classifier(new_data['model_weights'])

    # prediction after
    with torch.no_grad():
        prediction_after = net(new_data['model_weights']).detach().cpu().numpy()[0]

    return digit_net, new_data, prediction_before, prediction_after


def weight_reset(net, data, layers=None, neurons=None):
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prediction before
    with torch.no_grad():
        prediction_before = net(data['model_weights']).detach().cpu().numpy()[0]

    # copy data data
    new_weights = {}
    for k, v in data['model_weights'].items():
        new_weights[k] = v.clone()
    new_data = {'model_weights': new_weights, 'bias': data['bias'].clone()}

    for layer in new_data['model_weights']:
        if layers is None or layer in layers:
            if neurons is not None:
                weight_reset = new_data['model_weights'][layer].clone()
                kaiming_uniform_(weight_reset, a=math.sqrt(5))
                new_data['model_weights'][layer] = new_data['model_weights'][layer] * torch.logical_not(torch.from_numpy(neurons[layer]).to(device)) \
                                                   + weight_reset * torch.from_numpy(neurons[layer]).to(device)
            else:
                kaiming_uniform_(new_data['model_weights'][layer], a=math.sqrt(5))

    # create mnist classifier
    digit_net = get_digit_classifier(new_data['model_weights'])

    # prediction before
    with torch.no_grad():
        prediction_after = net(new_data['model_weights']).detach().cpu().numpy()[0]

    return digit_net, prediction_before, prediction_after


def dropout(net, data, prob=0.5, layers=None):
    net.eval()

    # prediction before
    with torch.no_grad():
        prediction_before = net(data['model_weights']).detach().cpu().numpy()[0]

    # create mnist classifier
    digit_net = get_digit_classifier(data['model_weights'])
    digit_net.add_dropout(prob, layers[0])

    return digit_net, prediction_before, prediction_before.copy()


def fine_tune(net, data, epochs=1, lr=1e-4, layers=None):
    net.eval()
    # prediction before
    with torch.no_grad():
        prediction_before = net(data['model_weights']).detach().cpu().numpy()[0]

    # create mnist classifier
    digit_net = get_digit_classifier(data['model_weights'])

    # load unbiased colored MNIST dataset
    mnist_dataset = load_unbiased_color_mnist_dataset(train=True)
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=1)

    digit_net.train()
    optimizer = torch.optim.Adam(digit_net.parameters(), lr=lr)
    loss_fcn = torch.nn.CrossEntropyLoss()
    for param in digit_net.parameters():
        param.requires_grad = False
    for name, layer in digit_net.named_children():
        if (layers[0] == "layer_0" or layers[0] == "layer_1") and name == "conv1":
            for param in layer.parameters():
                param.requires_grad = True
        elif (layers[0] == "layer_2" or layers[0] == "layer_3") and name == "conv2":
            for param in layer.parameters():
                param.requires_grad = True
        elif (layers[0] == "layer_4" or layers[0] == "layer_5") and name == "conv3":
            for param in layer.parameters():
                param.requires_grad = True
        elif (layers[0] == "layer_6" or layers[0] == "layer_7") and name == "dense1":
            for param in layer.parameters():
                param.requires_grad = True
        elif (layers[0] == "layer_8" or layers[0] == "layer_9") and name == "dense2":
            for param in layer.parameters():
                param.requires_grad = True

    for epoch in range(epochs):
        for X, y in mnist_dataloader:
            optimizer.zero_grad()
            pred = digit_net.forward(X)
            loss = loss_fcn(pred, y[:, 0].long())
            loss.backward()
            optimizer.step()

    for param in digit_net.parameters():
        param.requires_grad = True

    new_data = {'model_weights': digit_net.get_model_weights(), 'bias': data['bias'].clone()}

    # prediction before
    with torch.no_grad():
        prediction_after = net(new_data['model_weights']).detach().cpu().numpy()[0]

    return digit_net, prediction_before, prediction_after


# MAIN

def main():
    # load options
    with open('options/bias_mitigation_options.json', 'r') as f:
        options = json.load(f)

    # load network weights dataset
    weights_dataset = load_network_weights_dataset(options)
    weights_dataloader = DataLoader(weights_dataset, batch_size=1, shuffle=True)

    # load pre-trained network
    layer_shapes = []
    for _, arr in weights_dataset[0]['model_weights'].items():
        layer_shapes.append(arr.shape)
    net = load_network(options, layer_shapes)

    # bias metrics
    actual_bias = []
    metrics_before = {'prediction': []}
    metrics_after = {'prediction': []}
    changesPerSample = []
    for metric in options['bias_metrics']:
        metrics_before[metric] = []
        metrics_after[metric] = []
    count = 0
    for data in weights_dataloader:
        count += 1
        if count > options['num_samples']:
            break
        star_time = time.time()

        # save actual bias
        actual_bias.append(np.argmax(data['bias'][0].detach().cpu().numpy()))

        # test with unbiased coloredMNIST dataset before bias mitigation
        digit_net = get_digit_classifier(data['model_weights'])
        for metric in options['bias_metrics']:
            if metric == "unbiased_accuracy":
                accuracy = accuracy_on_unbiased_mnist(digit_net)
                metrics_before[metric].append(accuracy)
            elif metric == "score_ratio":
                ratio = score_ratio(digit_net)
                metrics_before[metric].append(ratio)
            else:
                raise ValueError("Bias metric [{}] not recognized.".format(metric))

        # ablation study (& bias mitigation)
        method = options['mitigation_method'] if options['detection_method'] is None else options['detection_method']
        if method['name'] == "fast gradient sign method":
            digit_net, new_data, prediction_before, prediction_after = fast_gradient_sign_method(net, data, epsilon=method['epsilon'], steps=method['steps'], layers=method['layers'])
        elif method['name'] == "gradient method":
            digit_net, new_data, prediction_before, prediction_after = gradient_method(net, data, epsilon=method['epsilon'], steps=method['steps'], layers=method['layers'])
        else:
            raise ValueError("Bias localization/mitigation method [{}] not recognized.".format(method['name']))
        if options['detection_method'] is None:
            metrics_before['prediction'].append(prediction_before)
            metrics_after['prediction'].append(prediction_after)
        changesPerLayer = computeAbsoluteChangesInSample(data, new_data)
        visualizeChangeInSample(changesPerLayer, count)
        changesPerSample.append(changesPerLayer)
        mitigation_layers = get_most_biased_layers(changesPerLayer, 1)
        mitigation_neurons = {}
        for layer in changesPerLayer:
            mitigation_neurons[layer] = changesPerLayer[layer] >= options['neuron_change_threshold']

        # bias mitigation
        if options['detection_method'] is not None and options['mitigation_method'] is not None:
            method = options['mitigation_method']
            if method['name'] == "fast gradient sign method":
                if options['mitigation_per_layer']:
                    digit_net, _, prediction_before, prediction_after = fast_gradient_sign_method(net, data, epsilon=method['epsilon'], steps=method['steps'], layers=mitigation_layers)
                else:
                    digit_net, _, prediction_before, prediction_after = fast_gradient_sign_method(net, data, epsilon=method['epsilon'], steps=method['steps'], neurons=mitigation_neurons)
            elif method['name'] == "gradient method":
                if options['mitigation_per_layer']:
                    digit_net, _, prediction_before, prediction_after = gradient_method(net, data, epsilon=method['epsilon'], steps=method['steps'], layers=mitigation_layers)
                else:
                    digit_net, _, prediction_before, prediction_after = gradient_method(net, data, epsilon=method['epsilon'], steps=method['steps'], neurons=mitigation_neurons)

            elif method['name'] == "weight reset":
                if options['mitigation_per_layer']:
                    digit_net, prediction_before, prediction_after = weight_reset(net, data, layers=mitigation_layers)
                else:
                    digit_net, prediction_before, prediction_after = weight_reset(net, data, neurons=mitigation_neurons)
            elif method['name'] == "dropout":
                digit_net, prediction_before, prediction_after = dropout(net, data, prob=method['prob'], layers=mitigation_layers)
            elif method['name'] == "fine-tune":
                digit_net, prediction_before, prediction_after = fine_tune(net, data, epochs=method['epochs'], lr=method['lr'], layers=mitigation_layers)
            else:
                raise ValueError("Bias mitigation method [{}] not recognized.".format(method['name']))
            metrics_before['prediction'].append(prediction_before)
            metrics_after['prediction'].append(prediction_after)

        # test with unbiased coloredMNIST dataset after bias mitigation
        for metric in options['bias_metrics']:
            if metric == "unbiased_accuracy":
                accuracy = accuracy_on_unbiased_mnist(digit_net)
                metrics_after[metric].append(accuracy)
            elif metric == "score_ratio":
                ratio = score_ratio(digit_net)
                metrics_after[metric].append(ratio)
            else:
                raise ValueError("Bias metric [{}] not recognized.".format(metric))

        print(f'Sample {count} / {options["num_samples"]}, Time: {time.time() - star_time:.3f}s')


    visualizeChanges(changesPerSample)
    if options['mitigation_method'] is not None:
        plotMetrics(metrics_before, metrics_after, actual_bias)

    iter = 1
    for pb, ab, pa, aa in zip(metrics_before['prediction'], metrics_before['unbiased_accuracy'],
                              metrics_after['prediction'], metrics_after['unbiased_accuracy']):
        print(f'Sample {iter}:')
        print(f'Accuracy: {ab:.5} ---> {aa:.5}')
        print(f'Prediction: {list(np.around(pb, decimals=5))} ---> {list(np.around(pa, decimals=5))}')
        iter += 1


def plotMetrics(metrics_before, metrics_after, actual_bias):
    """
    plot accuracy before mitigation and after mitigation, plots also mean prediction before and after mitigation
    """
    assert(metrics_before.keys() == metrics_after.keys())

    # plot predictions change
    pred_before_l = metrics_before['prediction']
    pred_after_l = metrics_after['prediction']
    pred_before = np.reshape(pred_before_l[0], (1,len(pred_before_l[0])))
    pred_after = np.reshape(pred_after_l[0], (1,len(pred_after_l[0])))
    for i in range(1, len(pred_before_l)):
        pred_before = np.append(pred_before, np.reshape(pred_before_l[i], (1,len(pred_before_l[i]))), axis=0)
        pred_after = np.append(pred_after, np.reshape(pred_after_l[i], (1,len(pred_after_l[i]))), axis=0)
    predictions = pd.DataFrame()
    classes = ['0.02','0.03','0.04','0.05']
    for i in range(0,len(classes)):
        predictions = predictions.append(
            {
                'category': classes[i],
                'before': np.mean(pred_before[:,i]),
                'after': np.mean(pred_after[:,i])
            }, ignore_index=True
        )
    predictions = predictions.set_index('category')
    fig = predictions.plot(kind="bar", title="prediction mean before and after mitigation", figsize=(10,10))
    fig.get_figure().savefig("plots/predictionAfterMitigation.jpg")

    # plot accuracy change
    biases = ['0.02', '0.03', '0.04', '0.05']
    actual_bias = np.array(actual_bias)
    accuracy = pd.DataFrame()
    for i, b in enumerate(biases):
        accuracy = accuracy.append({
                'category': b,
                'before': np.mean(np.array(metrics_before["unbiased_accuracy"])[np.array(actual_bias) == i]),
                'after': np.mean(np.array(metrics_after['unbiased_accuracy'])[np.array(actual_bias) == i])
                },
                ignore_index=True)
    accuracy = accuracy.set_index('category')
    fig = accuracy.plot(kind="bar", title="accuracy mean before and after mitigation", figsize=(10, 10))
    fig.get_figure().savefig("plots/accuracyAfterMitigation.jpg")


def get_most_biased_layers(changesPerLayer, num):
    mean_changes = []
    for layer, bias_layer in zip(list(changesPerLayer.keys())[::2], list(changesPerLayer.keys())[1::2]):
        mean_changes.append((layer, bias_layer, np.mean(np.concatenate((changesPerLayer[layer], changesPerLayer[bias_layer]), axis=None))))
    mean_changes.sort(key=lambda x: x[2])
    result = []
    for l1, l2, _ in mean_changes[-num:]:
        result.append(l1)
        result.append(l2)
    return result


def computeAbsoluteChangesInSample(data, new_data):
    """
    computes absolut weight change per neuron for every sample
    """
    changesPerLayer = {}
    for layer in data['model_weights']:
        assert (np.shape(data['model_weights'][layer]) == np.shape(new_data['model_weights'][layer]))
        absoluteChange = np.abs(data['model_weights'][layer].detach().cpu().numpy() - new_data['model_weights'][layer].detach().cpu().numpy())
        changesPerLayer[layer] = absoluteChange
    return changesPerLayer


def visualizeChangeInSample(changesPerLayer, sampleNumber):
    """
    plots absolut weight change per layer for every sample
    """
    # transform data to plot
    results = pd.DataFrame()
    for layer in changesPerLayer:
        absoluteChange = changesPerLayer[layer]
        results = results.append({
            'layer': layer,
            'mean': np.mean(absoluteChange),
            'median': np.median(absoluteChange),
            'max': np.max(absoluteChange),
            'min': np.min(absoluteChange)},
            ignore_index=True)
    results = results.set_index('layer')
    fig = results.plot(kind="bar", title="absolut weight changes per layer in sample " + str(sampleNumber),
                       figsize=(10, 10))
    fig.get_figure().savefig("ablationPlots/changesPerLayer" + str(sampleNumber) + ".jpg")


def visualizeChanges(changesPerSample):
    """
    plots mean of mean change per sampled layer and mean of mean change per sampled neuron (both on layer level)
    """
    # transform data to plot
    dictionary = {}
    sample = changesPerSample[0]
    for layer in sample:
        dictionary[layer] = np.reshape(sample[layer], (1, np.prod(sample[layer].shape)))

    for sample in changesPerSample[1:]:
        for layer in sample:
            transformed = np.reshape(sample[layer], (1, np.prod(sample[layer].shape)))
            dictionary[layer] = np.append(dictionary[layer], transformed, axis=0)

    results = pd.DataFrame()
    for layer in dictionary.keys():
        results = results.append({
            'layer': layer,
            'mean of mean per sampled layer': np.mean(np.mean(dictionary[layer], axis=1)),
            'mean of mean per sampled neuron': np.mean(np.mean(dictionary[layer], axis=0))},
            ignore_index=True)
    results = results.set_index('layer')
    fig = results.plot(kind="bar", title="absolut weight changes per layer over all samples", figsize=(10, 10))
    fig.get_figure().savefig("ablationPlots/changesOverall.jpg")


if __name__ == '__main__':
    main()

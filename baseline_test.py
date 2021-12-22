import numpy

from utils.model_operations import ModelDataset
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from dimension_reduction.pca_dimension_reduction_model import PCADimensionReductionModel
from dimension_reduction.kernel_pca_dimension_reduction_model import KernelPCADimensionReductionModel
from dimension_reduction.autoencoder_dimension_reduction_model import AutoencoderDimensionReductionModel
from utils.plots import plot_accuracy_matrix
import time


bias = ['0.02', '0.03', '0.04', '0.05']
bias_to_label = {'0.02': 3, '0.03': 2, '0.04': 1, '0.05': 0}


def load_weights():
    # load training data
    train = pd.DataFrame()
    for b in bias:
        model_data = ModelDataset(bias=b, data_directory='data/DigitWdb/train')
        for model_idx in range(len(model_data)):
            print(f"Train Model {model_idx}/{len(model_data)}")
            weights = model_data[model_idx].get_weights()
            for i in range(len(weights)):
                weights[i] = np.ravel(weights[i], order='F')
            weights = np.concatenate(weights)

            train = train.append({'model': model_idx, 'weights': weights, 'bias': b}, ignore_index=True)

    # load testing data
    test = pd.DataFrame()
    for b in bias:
        model_data = ModelDataset(bias=b, data_directory='data/DigitWdb/test')
        for model_idx in range(len(model_data)):
            print(f"Test Model {model_idx}/{len(model_data)}")
            weights = model_data[model_idx].get_weights()
            for i in range(len(weights)):
                weights[i] = np.ravel(weights[i], order='F')
            weights = np.concatenate(weights)

            test = test.append({'model': model_idx, 'weights': weights, 'bias': b},ignore_index=True)

    return train, test


def load_weights_per_layer():
    # load training data
    train = pd.DataFrame()
    for b in bias:
        model_data = ModelDataset(bias=b, data_directory='data/DigitWdb/train')
        for model_idx in range(len(model_data)):
            print(f"Train Model {model_idx}/{len(model_data)}")
            weights = model_data[model_idx].get_weights()
            for i in range(0, len(weights), 2):
                weight = np.concatenate([np.ravel(weights[i], order='F'), np.ravel(weights[i+1], order='F')])
                train = train.append({'model': model_idx, 'layer': int(i / 2), 'weights': weight, 'bias': b}, ignore_index=True)

    # load testing data
    test = pd.DataFrame()
    for b in bias:
        model_data = ModelDataset(bias=b, data_directory='data/DigitWdb/test')
        for model_idx in range(len(model_data)):
            print(f"Test Model {model_idx}/{len(model_data)}")
            weights = model_data[model_idx].get_weights()
            for i in range(0, len(weights), 2):
                weight = np.concatenate([np.ravel(weights[i], order='F'), np.ravel(weights[i + 1], order='F')])
                test = test.append({'model': model_idx, 'layer': int(i / 2), 'weights': weight, 'bias': b}, ignore_index=True)

    return train, test


def save_data_as_parquet():
    train, test = load_weights()
    train.to_parquet('data/train_weights.parquet', index=False)
    test.to_parquet('data/test_weights.parquet', index=False)

    train, test = load_weights_per_layer()
    train.to_parquet('data/train_layer_weights.parquet', index=False)
    test.to_parquet('data/test_layer_weights.parquet', index=False)


def load_data_from_parquet(per_layer=False):
    if per_layer:
        train = pd.read_parquet('data/train_layer_weights.parquet')
        test = pd.read_parquet('data/test_layer_weights.parquet')
    else:
        train = pd.read_parquet('data/train_weights.parquet')
        test = pd.read_parquet('data/test_weights.parquet')

    return train, test


def autoencoder_reduce_dimensions(dim=100):
    train, test = load_data_from_parquet()

    train_X = pd.DataFrame(np.array(list(train['weights'])))
    train_y = pd.DataFrame([bias_to_label[b] for b in train['bias']])
    test_X = pd.DataFrame(np.array(list(test['weights'])))
    test_y = pd.DataFrame([bias_to_label[b] for b in test['bias']])

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    dim_reduction = AutoencoderDimensionReductionModel(in_dim=train_X.shape[1], out_dim=dim)
    print(dim_reduction.model)
    train_X = dim_reduction.fit_transform(train_X, train_y)
    dim_reduction.save_model(name=f'autoencoder_{dim}')
    test_X = dim_reduction.transform(test_X)

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    if not os.path.exists(f'data/reduced/autoencoder_{dim}'):
        os.mkdir(f'data/reduced/autoencoder_{dim}')

    train_X.to_csv(f'data/reduced/autoencoder_{dim}/train_X.csv', index=False)
    train_y.to_csv(f'data/reduced/autoencoder_{dim}/train_y.csv', index=False)
    test_X.to_csv(f'data/reduced/autoencoder_{dim}/test_X.csv', index=False)
    test_y.to_csv(f'data/reduced/autoencoder_{dim}/test_y.csv', index=False)


def pca_reduce_dimensions(dim=100, kernel=None, **params):
    train, test = load_data_from_parquet()

    train_X = pd.DataFrame(np.array(list(train['weights'])))
    train_y = pd.DataFrame([bias_to_label[b] for b in train['bias']])
    test_X = pd.DataFrame(np.array(list(test['weights'])))
    test_y = pd.DataFrame([bias_to_label[b] for b in test['bias']])

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    if kernel is None:
        dim_reduction = PCADimensionReductionModel(dim, **params)
    else:
        dim_reduction = KernelPCADimensionReductionModel(dim, **params)

    train_X = dim_reduction.fit_transform(train_X)
    test_X = dim_reduction.transform(test_X)

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    if kernel is None:
        name = f'pca_{dim}'
    else:
        name = f'{kernel}_pca_{dim}'

    if not os.path.exists(f'data/reduced/{name}'):
        os.mkdir(f'data/reduced/{name}')

    train_X.to_csv(f'data/reduced/{name}/train_X.csv', index=False)
    train_y.to_csv(f'data/reduced/{name}/train_y.csv', index=False)
    test_X.to_csv(f'data/reduced/{name}/test_X.csv', index=False)
    test_y.to_csv(f'data/reduced/{name}/test_y.csv', index=False)


def autoencoder_reduce_dimensions_per_layer(dim=100):
    train, test = load_data_from_parquet(per_layer=True)

    for l in range(5):
        print("Layer", (l+1))
        train_X = pd.DataFrame(np.array(list(train['weights'].loc[train['layer'] == l])))
        train_y = pd.DataFrame([bias_to_label[b] for b in train['bias'].loc[train['layer'] == l]])
        test_X = pd.DataFrame(np.array(list(test['weights'].loc[train['layer'] == l])))
        test_y = pd.DataFrame([bias_to_label[b] for b in test['bias'].loc[train['layer'] == l]])

        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        dim_reduction = AutoencoderDimensionReductionModel(in_dim=train_X.shape[1], out_dim=dim)
        print(dim_reduction.model)
        train_X = dim_reduction.fit_transform(train_X)
        dim_reduction.save_model(name=f'autoencoder_{dim}_l{l+1}')
        test_X = dim_reduction.transform(test_X)

        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        if not os.path.exists(f'data/reduced/autoencoder_{dim}_l{l+1}'):
            os.mkdir(f'data/reduced/autoencoder_{dim}_l{l+1}')

        train_X.to_csv(f'data/reduced/autoencoder_{dim}_l{l+1}/train_X.csv', index=False)
        train_y.to_csv(f'data/reduced/autoencoder_{dim}_l{l+1}/train_y.csv', index=False)
        test_X.to_csv(f'data/reduced/autoencoder_{dim}_l{l+1}/test_X.csv', index=False)
        test_y.to_csv(f'data/reduced/autoencoder_{dim}_l{l+1}/test_y.csv', index=False)


def pca_reduce_dimensions_per_layer(dim=100, kernel=None, **params):
    train, test = load_data_from_parquet(per_layer=True)

    for l in range(5):
        print("Layer", (l + 1))
        train_X = pd.DataFrame(np.array(list(train['weights'].loc[train['layer'] == l])))
        train_y = pd.DataFrame([bias_to_label[b] for b in train['bias'].loc[train['layer'] == l]])
        test_X = pd.DataFrame(np.array(list(test['weights'].loc[train['layer'] == l])))
        test_y = pd.DataFrame([bias_to_label[b] for b in test['bias'].loc[train['layer'] == l]])

        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        if kernel is None:
            dim_reduction = PCADimensionReductionModel(dim, **params)
        else:
            dim_reduction = KernelPCADimensionReductionModel(dim, **params)

        train_X = dim_reduction.fit_transform(train_X)
        test_X = dim_reduction.transform(test_X)

        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        if kernel is None:
            name = f'pca_{dim}_l{l+1}'
        else:
            name = f'{kernel}_pca_{dim}_l{l+1}'

        if not os.path.exists(f'data/reduced/{name}'):
            os.mkdir(f'data/reduced/{name}')

        train_X.to_csv(f'data/reduced/{name}/train_X.csv', index=False)
        train_y.to_csv(f'data/reduced/{name}/train_y.csv', index=False)
        test_X.to_csv(f'data/reduced/{name}/test_X.csv', index=False)
        test_y.to_csv(f'data/reduced/{name}/test_y.csv', index=False)


def get_classification_accuracy(classifier, reduced_data='autoencoder_100', scaling=True, selected_features=None, selected_labels=None, plot=False):
    if isinstance(reduced_data, str):
        train_X = pd.read_csv(f'data/reduced/{reduced_data}/train_X.csv')
        train_y = np.array(pd.read_csv(f'data/reduced/{reduced_data}/train_y.csv'))[:, 0]
        test_X = pd.read_csv(f'data/reduced/{reduced_data}/test_X.csv')
        test_y = np.array(pd.read_csv(f'data/reduced/{reduced_data}/test_y.csv'))[:, 0]
    else:
        train_X = reduced_data['train_X']
        train_y = reduced_data['train_y']
        test_X = reduced_data['test_X']
        test_y = reduced_data['test_y']

    if selected_features is not None:
        train_X = train_X.iloc[:, selected_features]
        test_X = test_X.iloc[:, selected_features]

    if selected_labels is not None:
        train_keep = np.isin(train_y, selected_labels)
        train_X = train_X.loc[train_keep]
        train_y = train_y[train_keep]
        test_keep = np.isin(test_y, selected_labels)
        test_X = test_X.loc[test_keep]
        test_y = test_y[test_keep]

    if scaling:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

    classifier.fit(train_X, train_y)
    pred_y = classifier.predict(test_X)

    accuracy = accuracy_score(test_y, pred_y)
    # print("Accuracy:", accuracy)

    if plot:
        class_names = reversed(bias) if selected_labels is None else np.array(list(reversed(bias)))[selected_labels]
        plot_accuracy_matrix(pred_y, test_y, class_names=class_names)

    return accuracy


def get_classification_accuracy_per_layer(classifier, reduced_data='autoencoder_100', scaling=True, selected_features=None, selected_labels=None, plot=False):
    accuracies = []
    for l in range(5):
        train_X = pd.read_csv(f'data/reduced/{reduced_data}_l{l+1}/train_X.csv')
        train_y = np.array(pd.read_csv(f'data/reduced/{reduced_data}_l{l+1}/train_y.csv'))[:, 0]
        test_X = pd.read_csv(f'data/reduced/{reduced_data}_l{l+1}/test_X.csv')
        test_y = np.array(pd.read_csv(f'data/reduced/{reduced_data}_l{l+1}/test_y.csv'))[:, 0]

        if selected_features is not None:
            train_X = train_X.iloc[:, selected_features]
            test_X = test_X.iloc[:, selected_features]

        if selected_labels is not None:
            train_keep = np.isin(train_y, selected_labels)
            train_X = train_X.loc[train_keep]
            train_y = train_y[train_keep]
            test_keep = np.isin(test_y, selected_labels)
            test_X = test_X.loc[test_keep]
            test_y = test_y[test_keep]

        if scaling:
            scaler = StandardScaler()
            train_X = scaler.fit_transform(train_X)
            test_X = scaler.transform(test_X)

        classifier.fit(train_X, train_y)
        pred_y = classifier.predict(test_X)

        accuracy = accuracy_score(test_y, pred_y)
        print(f"Layer {l+1} Accuracy:", accuracy)
        accuracies.append(accuracy)

        if plot:
            plot_accuracy_matrix(pred_y, test_y, class_names=reversed(bias), title=f"Layer {l} Accuracy")

    if plot:
        plt.bar([0, 1, 2, 3, 4], accuracies)
        plt.title(f"Layer Accuracy Comparison")
        plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
        plt.ylabel('accuracy')
        plt.xlabel('layer')
        plt.show()
        plt.savefig(f'data/results/{type(classifier).__name__}_{reduced_data}_comparison_accuracy.png')


def get_two_best_components(classifier, reduced_data='autoencoder_100', num_features=100, selected_labels=None, save=False, filename="best_two_components"):
    best = (-1, -1)
    best_score = -1
    for i in range(num_features):
        for j in range(i, num_features):
            acc = get_classification_accuracy(classifier, reduced_data, scaling=True, selected_features=[i, j], selected_labels=selected_labels)
            if acc > best_score:
                best_score = acc
                best = (i, j)

    get_classification_accuracy(classifier, reduced_data, scaling=True, selected_features=list(best), selected_labels=selected_labels, plot=True)

    # X = pd.concat([pd.read_csv(f'data/reduced/{reduced_data}/train_X.csv'), pd.read_csv(f'data/reduced/{reduced_data}/test_X.csv')]).iloc[:, list(best)]
    # y = pd.concat([pd.read_csv(f'data/reduced/{reduced_data}/train_y.csv'), pd.read_csv(f'data/reduced/{reduced_data}/test_y.csv')]).iloc[:, 0]

    train_X = pd.read_csv(f'data/reduced/{reduced_data}/train_X.csv').iloc[:, list(best)]
    train_y = np.array(pd.read_csv(f'data/reduced/{reduced_data}/train_y.csv'))[:, 0]
    test_X = pd.read_csv(f'data/reduced/{reduced_data}/test_X.csv').iloc[:, list(best)]
    test_y = np.array(pd.read_csv(f'data/reduced/{reduced_data}/test_y.csv'))[:, 0]

    if selected_labels is not None:
        train_keep = np.isin(train_y, selected_labels)
        train_X = train_X.loc[train_keep]
        train_y = train_y[train_keep]
        test_keep = np.isin(test_y, selected_labels)
        test_X = test_X.loc[test_keep]
        test_y = test_y[test_keep]

    scatter = plt.scatter(train_X.iloc[:, 0], train_X.iloc[:, 1], s=4, c=train_y, marker='o', cmap='gist_rainbow')
    plt.scatter(test_X.iloc[:, 0], test_X.iloc[:, 1], s=4, c=test_y, marker='x', cmap='gist_rainbow')
    plt.legend(*scatter.legend_elements())
    plt.xlabel(str(best[0]))
    plt.ylabel(str(best[1]))
    plt.show()
    if save:
        plt.savefig(f"results/{filename}.png")

def get_best_components_and_accuracy(classifier, reduced_data='autoencoder_100', num_features=100, selected_labels=None):
    train_X = pd.read_csv(f'data/reduced/{reduced_data}/train_X.csv')
    train_y = np.array(pd.read_csv(f'data/reduced/{reduced_data}/train_y.csv'))[:, 0]

    data = {
        'train_X': train_X.loc[[i for i in range(len(train_X)) if (i+1) % 4 != 0]],
        'train_y': train_y[[i for i in range(len(train_y)) if (i + 1) % 4 != 0]],
        'test_X': train_X.loc[[i for i in range(len(train_X)) if (i + 1) % 4 == 0]],
        'test_y': train_y[[i for i in range(len(train_y)) if (i + 1) % 4 == 0]]
    }

    features = []
    accs = []
    best_acc = -2
    best_score = -1
    start_time = time.time()
    while best_acc < best_score:
        print(time.time() - start_time)
        best_acc = best_score
        accs.append(best_acc)
        best = -1
        best_score = -1
        for i in range(num_features):
            if i not in features:
                feats = features.copy()
                feats.append(i)
                acc = get_classification_accuracy(classifier, reduced_data, scaling=True, selected_features=feats, selected_labels=selected_labels)
                if acc > best_score:
                    best_score = acc
                    best = i
        if best_acc < best_score:
            features.append(best)

    acc = get_classification_accuracy(classifier, reduced_data, scaling=True, selected_features=features, selected_labels=selected_labels)

    return acc, accs, features, time.time() - start_time



from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

if __name__ == '__main__':
    models = []
    models.append(("svc poly 3", SVC(kernel="poly", degree=3, class_weight="balanced")))
    models.append(("svc rbf", SVC(C=0.5, kernel="rbf", class_weight="balanced")))
    models.append(("svc sigmoid", SVC(C=3, kernel="sigmoid", class_weight="balanced")))
    models.append(("svc linear l1", LinearSVC(class_weight="balanced", penalty='l1', dual=False)))
    models.append(("svc linear l2", LinearSVC(class_weight="balanced", penalty='l2')))
    models.append(("bernoulli naive bayes", BernoulliNB()))
    models.append(("gaussian naive bayes", GaussianNB()))
    models.append(("decision tree", DecisionTreeClassifier(class_weight="balanced")))
    models.append(("extra tree", ExtraTreeClassifier(class_weight="balanced")))
    models.append(("extra trees", ExtraTreesClassifier(n_estimators=100, class_weight="balanced", n_jobs=4)))
    models.append(("k-nearest-neighbors distance", KNeighborsClassifier(weights="distance", n_jobs=4)))
    models.append(("k-nearest-neighbors uniform", KNeighborsClassifier(weights="uniform", n_jobs=4)))
    models.append(("linear dicriminant analysis", LinearDiscriminantAnalysis()))
    models.append(("nearest centroid", NearestCentroid()))
    models.append(("quadratic discriminant analysis", QuadraticDiscriminantAnalysis()))
    models.append(("random forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=4)))
    models.append(("ridge classification", RidgeClassifier(class_weight="balanced")))
    models.append(("perceptron", Perceptron(class_weight="balanced")))




    # model = ExtraTreesClassifier(n_estimators=100)
    # model = LinearSVC()

    for name, model in models:
        # acc, accs, feat, runtime = get_best_components_and_accuracy(model, reduced_data='pca_100', num_features=100, selected_labels=[0,1,2,3])
        # print(name, "\t", acc, "\t", f"{runtime/60:.2f}", "\t", len(feat))
        # print(accs)
        # print(feat)
        # print()
        acc = get_classification_accuracy(model, reduced_data="pca_100", selected_labels=[0,3], plot=False)
        print(name, "\t", acc)
        print()

    # best = 0
    # for hls in range(50, 1050, 50):
    #     print(hls)
    #     model = ExtraTreeClassifier(class_weight="balanced")
    #     accuracy = get_classification_accuracy(model, reduced_data="pca_100", selected_labels=[0,1,2,3], plot=False)
    #     if accuracy > best:
    #         best = accuracy
    # print()
    # print("Best:", best)


    # for n in range(1, 200):
    #     for weights in ['uniform', 'distance']:
    #         print(n, weights)

    #         model = KNeighborsClassifier(n_neighbors=n, weights=weights)
    #         get_classification_accuracy(model, reduced_data="pca_10", selected_labels=[0, 3], plot=False)

    # get_two_best_components(model, reduced_data="pca_100", num_features=100, selected_labels=[0, 3], save=True, filename="two_best_pca_components_linear")
    # get_two_best_components(model, reduced_data="autoencoder_100", num_features=100, save=True)
    #
    # for l in range(1, 6):
    #     get_two_best_components(model, reduced_data=f"pca_100_l{l}", num_features=100, save=True)
    #     get_two_best_components(model, reduced_data=f"autoencoder_100_l{l}", num_features=100, save=True)

# fix baseline
# write about idea (feature mitigation)

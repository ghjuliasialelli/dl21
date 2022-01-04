import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_matrix(pred_y, true_y, class_names=None, title="Accuracy", save=False, filename='accuracy'):
    unique = np.sort(np.unique(true_y))

    accuracy = np.sum(pred_y == true_y) / len(true_y)

    acc_table = np.zeros((len(unique), len(unique)))
    for i in range(len(unique)):
        for j in range(len(unique)):
            acc_table[i, j] = np.sum(pred_y[true_y == unique[i]] == unique[j])
        acc_table[i] /= np.sum(true_y == unique[i])

    fig, ax = plt.subplots()
    for i in range(len(unique)):
        for j in range(len(unique)):
            ax.text(j, i, f"{acc_table[i, j]:.3f}", ha='center', va='center', color='blue')
    ax.set_title(f"{title} ({accuracy:.5f})")
    ax.xaxis.tick_top()
    if class_names is not None:
        plt.xticks(np.arange(0, len(unique)), class_names)
        plt.yticks(np.arange(0, len(unique)), class_names)
    else:
        plt.xticks(np.arange(0, len(unique)), unique)
        plt.yticks(np.arange(0, len(unique)), unique)
    plt.ylabel("ground truth")
    plt.xlabel("prediction")
    plt.imshow(acc_table, cmap='Wistia', interpolation='nearest')
    plt.show()
    if save:
        plt.savefig(f"results/{filename}.png")
    plt.close(fig)

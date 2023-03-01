import matplotlib.pyplot as plt
import numpy as np


def plot(labels, prediction, train_size):
    data_size = len(labels)
    plt.figure(figsize=(40, 6.5))
    plt.scatter(np.arange(data_size), labels, c='r', label='real_data', s=0.7)
    plt.scatter(np.arange(train_size),
                prediction[0:train_size], c='b', label='already_trained', s=0.7)
    plt.scatter(np.arange(train_size, data_size),
                prediction[train_size:], c='y', label='never_trained', s=0.7)
    plt.axvline(x=train_size, c='r', ls='--')
    plt.legend(loc='upper right')
    plt.show()

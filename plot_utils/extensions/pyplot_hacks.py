import numpy as np
import matplotlib.pyplot as plt


def bar_plot(X, xlabel, ylabel, facecolor=[0.5, 0.5, 0.5], xticks_labels=None, xticks_rotation=90, figsize=(5, 2.5)):
    fig = plt.figure(figsize=figsize)
    plt.gca().yaxis.grid(True)
    x_vec = np.arange(0, X.shape[0])
    #x_ticks = np.argsort(hist)[::-1]
    plt.bar(x_vec - 0.5, X, facecolor=facecolor)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks_labels is not None:
        plt.xticks(x_vec, xticks_labels, rotation=xticks_rotation)
    plt.xlim([-0.5, x_vec[-1] + 0.5])
    plt.tight_layout()


def histogram_plot(X, xlabel, ylabel, facecolor=[0.5, 0.5, 0.5], figsize=(5, 2.5)):
    fig = plt.figure(figsize=figsize)
    plt.grid()
    xlabel = "feature index"
    hist_data = np.concatenate(X)
    hist, bin_edges = np.histogram(hist_data, X.shape[1])
    x_vec = np.arange(0, X.shape[1])
    x_ticks = np.argsort(hist)[::-1]
    plt.bar(x_vec - 0.5, hist, facecolor=facecolor)
    # plt.hist(hist_data, bins=range(0, X.shape[1])) # +1 for plotting purposes
    # n, bins, patches = plt.hist(hist, bins=X.shape[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x_vec, x_ticks, rotation=90)
    plt.xlim([-0.5, x_vec[-1] + 0.5])
    plt.tight_layout()
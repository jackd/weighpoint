from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def _plot_confusion_matrix(
        ax, cm, cmap, normalize, target_names, display_counts, title):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if target_names is None:
        target_names = [str(i) for i in range(cm.shape[0])]

    # if target_names is not None:
    #     tick_marks = np.arange(len(target_names))
    #     ax.xticks(tick_marks, target_names, rotation=45)
    #     ax.yticks(tick_marks, target_names)
    totals = np.sum(cm, axis=1)

    if normalize:
        cmv = cm.astype('float') / np.expand_dims(totals, axis=1)
    else:
        cmv = cm

    ax.imshow(cmv, interpolation='nearest', cmap=cmap)

    if display_counts:
        # thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            if normalize:
                thresh = totals[i] / 2
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=6)
    ax.set_ylabel('True label')
    total = np.sum(totals)
    correct = np.sum(np.diag(cm))
    incorrect = total - correct
    ax.set_xlabel(
        'Predicted label\naccuracy={:0.4f}; misclass={:0.4f};'
        '\ncorrect={:d}; incorrect={:d}; total={:d}'.format(
            accuracy, misclass, correct, incorrect, total))
    ax.title.set_text(title)


def plot_confusion_matrices(cms, titles,
                            target_names=None,
                            title='Confusion matrix',
                            cmap=None,
                            normalize=True,
                            display_counts=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Originally from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    n_matrices = len(cms)

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig, axs = plt.subplots(1, n_matrices)
    for (ax, cm, title) in zip(axs, cms, titles):
        _plot_confusion_matrix(
            ax, cm, cmap, normalize, target_names, display_counts, title)

    plt.title(title)
    plt.tight_layout()
    plt.show()

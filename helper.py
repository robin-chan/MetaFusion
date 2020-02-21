#!/usr/bin/env python3
"""
script including
functions for easy usage in main scripts
"""

import numpy as np
import sys
from scipy.interpolate import interp1d
from global_defs import CONFIG
from in_out import metrics_load, metrics_dump


def concatenate_metrics(rule, num_imgs, save=False):
    metrics = metrics_load(0, rule)
    start = list([0, len(metrics["S"])])
    for i in range(1, num_imgs):
        # sys.stdout.write("\t concatenated file number {} / {}\r".format(i + 1, num_imgs))
        m = metrics_load(i, rule)
        start += [start[-1] + len(m["S"])]
        for j in metrics:
            metrics[j] += m[j]
    # print(" ")
    # print("connected components:", len(metrics['iou']))
    # print("non-empty connected components:", np.sum(np.asarray(metrics['S_in']) != 0))
    if (save == True):
        metrics_dump(metrics, "_all")
        metrics_dump(start, "_start")
    return metrics, start


def metrics_to_nparray(metrics, names, normalize=False, non_empty=False, all_metrics=[]):
    I = range(len(metrics['S_in']))
    if non_empty == True:
        I = np.asarray(metrics['S_in']) > 0
    M = np.asarray([np.asarray(metrics[m])[I] for m in names])
    if all_metrics == []:
        MM = M.copy()
    else:
        MM = np.asarray([np.asarray(all_metrics[m])[I] for m in names])
    if normalize == True:
        for i in range(M.shape[0]):
            if names[i] != "class":
                M[i] = (np.asarray(M[i]) - np.mean(MM[i], axis=-1)) / (np.std(MM[i], axis=-1) + 1e-10)
    M = np.squeeze(M.T)
    return M


def label_as_onehot(label, num_classes, shift_range=0):
    y = np.zeros((num_classes, label.shape[0], label.shape[1]))
    for c in range(shift_range, num_classes + shift_range):
        y[c - shift_range][label == c] = 1
    y = np.transpose(y, (1, 2, 0))  # shape is (height, width, num_classes)
    return y.astype('uint8')


def classes_to_categorical(classes, nc=None):
    classes = np.squeeze(np.asarray(classes))
    if nc == None:
        nc = np.max(classes)
    classes = label_as_onehot(classes.reshape((classes.shape[0], 1)), nc).reshape((classes.shape[0], nc))
    names = ["C_" + str(i) for i in range(nc)]
    return classes, names


def metrics_to_dataset(metrics, nclasses, non_empty=True, all_metrics=[]):
    X_names = sorted([m for m in metrics if m not in ["class", "iou", "iou0", "ID"] and "cprob" not in m])
    class_names = ["cprob" + str(i) for i in range(nclasses) if "cprob" + str(i) in metrics]
    Xa = metrics_to_nparray(metrics, X_names, normalize=True, non_empty=non_empty, all_metrics=all_metrics)
    classes = metrics_to_nparray(metrics, class_names, normalize=True, non_empty=non_empty, all_metrics=all_metrics)
    ya = metrics_to_nparray(metrics, ["iou"], normalize=False, non_empty=non_empty)
    y0a = metrics_to_nparray(metrics, ["iou0"], normalize=False, non_empty=non_empty)
    return Xa, classes, ya, y0a, X_names, class_names


def get_lambdas(n_steps, min_pow, max_pow):
    m = interp1d([0, n_steps - 1], [min_pow, max_pow])
    lambdas = [10 ** m(i).item() for i in range(n_steps)]
    return lambdas
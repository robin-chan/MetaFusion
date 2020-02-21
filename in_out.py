#!/usr/bin/env python3
"""
script including
functions for handling input/output like loading/saving
"""

import numpy as np
import os
import pickle
import h5py
import labels as labels
from PIL import Image

from global_defs import CONFIG


def get_save_path_input_i(i):
    return CONFIG.INPUT_DIR + "input_" + str(i) + ".hdf5"


def get_save_path_metrics_i(i, rule):
    return CONFIG.METRICS_DIR + rule + "/" + "metrics" + str(i) + ".p"


def get_save_path_components_i(i, rule):
    return CONFIG.COMPONENTS_DIR + rule + "/" + "components" + str(i) + ".p"


def get_save_path_prediction_i(i, rule):
    return CONFIG.PRED_DIR + rule + "/" + "labelTrainIds_pred" + str(i) + ".png"


def load_priors():
    priors = np.load(CONFIG.PRIORS_DIR + "priors-smooth-cat.npy")
    h, w, _ = priors.shape
    tmp = np.zeros([h, w, 19])
    for k in range(tmp.shape[-1]):
        tmp[:, :, k] += priors[:, :, labels.cs_trainId2Id[k].catId]
    priors = tmp + 1e-5
    return priors


def probs_gt_load(i):
    f_probs = h5py.File(get_save_path_input_i(i), "r")
    probs = np.asarray(f_probs['probabilities'])
    gt = np.asarray(f_probs['ground_truths'])
    probs = np.squeeze(probs)
    gt = np.squeeze(gt)
    return probs, gt, f_probs['image_path'][0].decode("utf8")


def metrics_load(i, rule="bayes"):
    read_path = get_save_path_metrics_i(i, rule)
    metrics = pickle.load(open(read_path, "rb"))
    return metrics


def components_load(i, rule="bayes"):
    read_path = get_save_path_components_i(i, rule)
    components = pickle.load(open(read_path, "rb"))
    return components


def prediction_load(i, rule="bayes"):
    read_path = get_save_path_prediction_i(i, rule)
    prediction = np.array(Image.open(read_path))
    return prediction


def metrics_dump(metrics, i, rule="bayes"):
    dump_path = get_save_path_metrics_i(i, rule)
    dump_dir = os.path.dirname(dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    pickle.dump(metrics, open(dump_path, "wb"))


def components_dump(components, i, rule="bayes"):
    dump_path = get_save_path_components_i(i, rule)
    dump_dir = os.path.dirname(dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    pickle.dump(components.astype('int16'), open(dump_path, "wb"))


def prediction_dump(pred, i, rule="bayes"):
    dump_path = get_save_path_prediction_i(i, rule)
    dump_dir = os.path.dirname(dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    Image.fromarray(pred.astype('uint8')).save(dump_path)


# def get_img_path_fname(filename):
#     path = []
#     for root, dirnames, filenames in os.walk(CONFIG.IMG_DIR):
#         for fn in filenames:
#             if filename in fn:
#                 path = os.path.join(root, fn)
#                 break
#     if path == []:
#         print("file", filename, "not found.")
#     return path


def stats_dump(stats, df_all, y0a):
    df_full = df_all.copy().loc[df_all['S_in'].nonzero()[0]]
    iou_corrs = df_full.corr()["iou"]
    mean_stats = dict({})
    std_stats = dict({})
    for s in stats:
        if s not in ["alphas", "n_av", "n_metrics", "metric_names"]:
            mean_stats[s] = np.mean(stats[s], axis=0)
            std_stats[s] = np.std(stats[s], axis=0)
    best_pen_ind = np.argmax(mean_stats['penalized_val_acc'])
    best_plain_ind = np.argmax(mean_stats['plain_val_acc'])

    # dump stats latex ready
    with open(CONFIG.RESULTS_DIR + 'av_results.txt', 'wt') as f:

        print(iou_corrs, file=f)
        print(" ", file=f)

        print("classification", file=f)
        print("                             & train                &  val                 &    \\\\ ", file=f)
        M = sorted([s for s in mean_stats if 'penalized' in s and 'acc' in s])
        print("ACC penalized               ", end=" & ", file=f)
        for s in M: print("${:.2f}\%".format(100 * mean_stats[s][best_pen_ind]) + "(\pm{:.2f}\%)$".format(
            100 * std_stats[s][best_pen_ind]), end=" & ", file=f)
        print("   \\\\ ", file=f)
        M = sorted([s for s in mean_stats if 'plain' in s and 'acc' in s])
        print("ACC unpenalized             ", end=" & ", file=f)
        for s in M: print("${:.2f}\%".format(100 * mean_stats[s][best_pen_ind]) + "(\pm{:.2f}\%)$".format(
            100 * std_stats[s][best_pen_ind]), end=" & ", file=f)
        print("   \\\\ ", file=f)
        M = sorted([s for s in mean_stats if 'entropy' in s and 'acc' in s])
        print("ACC entropy baseline        ", end=" & ", file=f)
        for s in M: print("${:.2f}\%".format(100 * mean_stats[s]) + "(\pm{:.2f}\%)$".format(100 * std_stats[s]),
                          end=" & ", file=f)
        print("   \\\\ ", file=f)

        M = sorted([s for s in mean_stats if 'penalized' in s and 'auroc' in s])
        print("AUROC penalized             ", end=" & ", file=f)
        for s in M: print("${:.2f}\%".format(100 * mean_stats[s][best_pen_ind]) + "(\pm{:.2f}\%)$".format(
            100 * std_stats[s][best_pen_ind]), end=" & ", file=f)
        print("   \\\\ ", file=f)
        M = sorted([s for s in mean_stats if 'plain' in s and 'auroc' in s])
        print("AUROC unpenalized           ", end=" & ", file=f)
        for s in M: print("${:.2f}\%".format(100 * mean_stats[s][best_pen_ind]) + "(\pm{:.2f}\%)$".format(
            100 * std_stats[s][best_pen_ind]), end=" & ", file=f)
        print("   \\\\ ", file=f)
        M = sorted([s for s in mean_stats if 'entropy' in s and 'auroc' in s])
        print("AUROC entropy baseline      ", end=" & ", file=f)
        for s in M: print("${:.2f}\%".format(100 * mean_stats[s]) + "(\pm{:.2f}\%)$".format(100 * std_stats[s]),
                          end=" & ", file=f)
        print("   \\\\ ", file=f)

        print(" ", file=f)
        print("regression", file=f)

        M = sorted([s for s in mean_stats if 'regr' in s and 'mse' in s and 'entropy' not in s])
        print("$\sigma$, all metrics       ", end=" & ", file=f)
        for s in M: print("${:.3f}".format(mean_stats[s]) + "(\pm{:.3f})$".format(std_stats[s]), end="    & ", file=f)
        print("   \\\\ ", file=f)
        M = sorted([s for s in mean_stats if 'regr' in s and 'mse' in s and 'entropy' in s])
        print("$\sigma$, entropy baseline  ", end=" & ", file=f)
        for s in M: print("${:.3f}".format(mean_stats[s]) + "(\pm{:.3f})$".format(std_stats[s]), end="    & ", file=f)
        print("   \\\\ ", file=f)

        M = sorted([s for s in mean_stats if 'regr' in s and 'r2' in s and 'entropy' not in s])
        print("$R^2$, all metrics          ", end=" & ", file=f)
        for s in M: print("${:.2f}\%".format(100 * mean_stats[s]) + "(\pm{:.2f}\%)$".format(100 * std_stats[s]),
                          end=" & ", file=f)
        print("   \\\\ ", file=f)
        M = sorted([s for s in mean_stats if 'regr' in s and 'r2' in s and 'entropy' in s])
        print("$R^2$, entropy baseline     ", end=" & ", file=f)
        for s in M: print("${:.2f}\%".format(100 * mean_stats[s]) + "(\pm{:.2f}\%)$".format(100 * std_stats[s]),
                          end=" & ", file=f)
        print("   \\\\ ", file=f)

        print(" ", file=f)
        M = sorted([s for s in mean_stats if 'iou' in s])
        for s in M: print(s, ": {:.0f}".format(mean_stats[s]) + "($\pm${:.0f})".format(std_stats[s]), file=f)
        print("IoU=0:", np.sum(y0a == 1), "of", y0a.shape[0], "non-empty components", file=f)
        print("IoU>0:", np.sum(y0a == 0), "of", y0a.shape[0], "non-empty components", file=f)
        print("total number of components: ", len(df_all), file=f)
        print(" ", file=f)

        dump_dir = CONFIG.STATS_DIR
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        pickle.dump(stats, open(dump_dir + "stats.p", "wb"))
    return mean_stats, std_stats

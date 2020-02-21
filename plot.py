#!/usr/bin/env python3
"""
script including
functions for visualizations
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr, kde

from global_defs import CONFIG
from in_out import get_save_path_input_i, probs_gt_load, components_load
import labels as labels

if CONFIG.DATASET == CONFIG.datasets[0]:
    trainId2label = {label.trainId: label for label in reversed(labels.cs_labels)}

elif CONFIG.DATASET == CONFIG.datasets[1]:
    trainId2label = {label.trainId: label for label in reversed(labels.ds20k_labels)}


os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'  # for tex in matplotlib
plt.rc('font', size=10, family='serif')
plt.rc('axes', titlesize=10)
plt.rc('figure', titlesize=10)
plt.rc('text', usetex=True)


def visualize_segments(comp, metric):
    R = np.asarray(metric)
    R = 1 - 0.5 * R
    G = np.asarray(metric)
    B = 0.3 + 0.35 * np.asarray(metric)

    R = np.concatenate((R, np.asarray([0, 1])))
    G = np.concatenate((G, np.asarray([0, 1])))
    B = np.concatenate((B, np.asarray([0, 1])))

    components = np.asarray(comp.copy(), dtype='int16')
    components[components < 0] = len(R) - 1
    components[components == 0] = len(R)

    img = np.zeros(components.shape + (3,))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x, y, 0] = R[components[x, y] - 1]
            img[x, y, 1] = G[components[x, y] - 1]
            img[x, y, 2] = B[components[x, y] - 1]

    img = np.asarray(255 * img).astype('uint8')

    return img


def visualize_regression_prediction_i(iou, iou_pred, i):
    if os.path.isfile(get_save_path_input_i(i)):

        probs, gt, path = probs_gt_load(i)
        input_image = Image.open(path).convert("RGB")
        input_image = np.asarray(input_image.resize(probs.shape[:2][::-1]))
        components = components_load(i)

        pred = np.asarray(np.argmax(probs, axis=-1), dtype='int')
        gt[gt == 255] = 0
        predc = np.asarray(
            [trainId2label[pred[p, q]].color for p in range(pred.shape[0]) for q in range(pred.shape[1])])
        gtc = np.asarray([trainId2label[gt[p, q]].color for p in range(gt.shape[0]) for q in range(gt.shape[1])])
        predc = predc.reshape(input_image.shape)
        gtc = gtc.reshape(input_image.shape)

        img_iou = visualize_segments(components, iou)

        I4 = predc / 2.0 + input_image / 2.0
        I3 = gtc / 2.0 + input_image / 2.0

        img_pred = visualize_segments(components, iou_pred)
        img = np.concatenate((img_iou, img_pred), axis=1)
        img2 = np.concatenate((I3, I4), axis=1)
        img = np.concatenate((img, img2), axis=0)
        image = Image.fromarray(img.astype('uint8'), 'RGB')

        seg_dir = CONFIG.IOU_SEG_VIS_DIR
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)
        image.save(seg_dir + "img" + str(i) + ".png")
        plt.close()

        print("stored:", seg_dir + "img" + str(i) + ".png")


def plot_roc_curve(Y, probs, roc_path):
    fpr, tpr, _ = roc_curve(Y, probs)
    roc_auc = auc(fpr, tpr)
    print("auc", roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of meta classification performance')
    plt.legend(loc="lower right")

    roc_dir = os.path.dirname(roc_path)
    if not os.path.exists(roc_dir):
        os.makedirs(roc_dir)

    plt.savefig(roc_path)
    print("roc curve saved to " + roc_path)
    plt.close()

    return roc_auc


def plot_regression(X2_val, y2_val, y2_pred, X_names):
    cmap = plt.get_cmap('tab20')
    plt.figure(figsize=(3, 3), dpi=300)
    plt.clf()
    S_ind = 0
    for S_ind in range(len(X_names)):
        if X_names[S_ind] == "S":
            break

    sizes = np.squeeze(X2_val[:, S_ind] * np.std(X2_val[:, S_ind]))
    sizes = sizes - np.min(sizes)
    sizes = sizes / np.max(sizes) * 50  # + 1.5
    x = np.arange(0., 1, .01)
    plt.plot(x, x, color='black', alpha=0.5, linestyle='dashed')
    plt.scatter(y2_val, np.clip(y2_pred, 0, 1), s=sizes, linewidth=.5, c=cmap(0), edgecolors=cmap(1), alpha=0.25)
    plt.xlabel('$\mathit{IoU}_\mathrm{adj}$')
    plt.ylabel('predicted $\mathit{IoU}_\mathrm{adj}$')
    plt.savefig(CONFIG.RESULTS_DIR + 'regression.png', bbox_inches='tight')

    plt.clf()


def plot_classif_hist(ya_val, ypred):
    figsize = (8.75, 5.25)
    plt.clf()

    density1 = kde.gaussian_kde(ya_val[ypred == 1])
    density2 = kde.gaussian_kde(ya_val[ypred == 0])

    density1.set_bandwidth(bw_method=density1.factor / 2.)
    density2.set_bandwidth(bw_method=density2.factor / 2.)

    x = np.arange(0., 1, .01)

    plt.clf()
    plt.figure(figsize=figsize)
    plt.plot(x, density1(x), color='red', alpha=0.66, label="pred. $IoU = 0$")
    plt.plot(x, density2(x), color='blue', alpha=0.66, label="pred. $IoU > 0$")
    plt.hist(ya_val[ypred == 1], bins=20, color='red', alpha=0.1, density=True)
    plt.hist(ya_val[ypred == 0], bins=20, color='blue', alpha=0.1, density=True)
    plt.legend(loc='upper right')
    plt.xlabel('$\mathit{IoU}_\mathrm{adj}$')
    plt.savefig(CONFIG.RESULTS_DIR + 'classif_hist.pdf', bbox_inches='tight')

    plt.clf()


def plot_classif(stats, mean_stats, X_names, class_names):
    nc = len(X_names) - len(class_names)
    coefs = np.squeeze(stats['coefs'][0, :, :])
    classcoefs = np.squeeze(stats['coefs'][0, :, nc:])
    coefs = np.concatenate([coefs[:, 0:nc], np.max(np.abs(coefs[:, nc:]), axis=1).reshape((coefs.shape[0], 1))], axis=1)
    max_acc = np.argmax(stats['penalized_val_acc'][0], axis=-1)
    lambdas = stats["lambdas"]

    cmap = plt.get_cmap('tab20')
    figsize = (8.75, 5.25)

    plt.clf()
    plt.semilogx(lambdas, stats['plain_val_acc'][0], label="unpenalized model", color=cmap(2))
    plt.semilogx(lambdas, stats['penalized_val_acc'][0], label="penalized model", color=cmap(0))
    plt.semilogx(lambdas, mean_stats['entropy_val_acc'] * np.ones((len(lambdas),)), label="entropy baseline",
                 color='black', linestyle='dashed')
    ymin, ymax = plt.ylim()
    plt.vlines(lambdas[max_acc], ymin, ymax, linestyle='dashed', linewidth=0.5, color='grey')
    legend = plt.legend(loc='lower right')
    plt.xlabel('$\lambda^{-1}$')
    plt.ylabel('classification accuracy')
    plt.axis('tight')
    plt.savefig(CONFIG.RESULTS_DIR + 'classif_perf.pdf', bbox_inches='tight')

    plt.clf()
    plt.semilogx(lambdas, stats['plain_val_auroc'][0], label="unpenalized model", color=cmap(2))
    plt.semilogx(lambdas, stats['penalized_val_auroc'][0], label="penalized model", color=cmap(0))
    plt.semilogx(lambdas, mean_stats['entropy_val_auroc'] * np.ones((len(lambdas),)), label="entropy baseline",
                 color='black', linestyle='dashed')
    ymin, ymax = plt.ylim()
    plt.vlines(lambdas[max_acc], ymin, ymax, linestyle='dashed', linewidth=0.5, color='grey')
    legend = plt.legend(loc='lower right')
    plt.xlabel('$\lambda^{-1}$')
    plt.ylabel('AUROC')
    plt.axis('tight')
    plt.savefig(CONFIG.RESULTS_DIR + 'classif_auroc.pdf', bbox_inches='tight')
    plt.close()


def add_scatterplot_vs_iou(ious, sizes, dataset, shortname, size_fac, scale, setylim=True):
    cmap = plt.get_cmap('tab20')
    rho = pearsonr(ious, dataset)
    plt.title(r"$\rho = {:.05f}$".format(rho[0]))
    plt.scatter(ious, dataset, s=sizes / np.max(sizes) * size_fac, linewidth=.5, c=cmap(0), edgecolors=cmap(1),
                alpha=.25)
    plt.xlabel('$\mathit{IoU}_\mathrm{adj}$', labelpad=-10)
    plt.ylabel(shortname, labelpad=-8)
    plt.ylim(-.05, 1.05)
    plt.xticks((0, 1), fontsize=10 * scale)
    plt.yticks((0, 1), fontsize=10 * scale)


def plot_scatter(df_full, m='E'):
    print("")
    print("making iou scatterplot ...")
    scale = .75
    size_fac = 50 * scale

    plt.figure(figsize=(3, 3), dpi=300)
    add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full[m], m, size_fac, scale)

    plt.tight_layout(pad=1.0 * scale, w_pad=0.5 * scale, h_pad=1.5 * scale)
    save_path = os.path.join(CONFIG.RESULTS_DIR, 'iou_vs_' + m + '.png')
    plt.savefig(save_path, bbox_inches='tight')
    print("scatterplots saved to " + save_path)
    plt.close()


def save_prediction_mask(pred, i, dirname):
    predc = np.asarray([trainId2label[pred[p, q]].color for p in range(pred.shape[0]) for q in range(pred.shape[1])])
    predc = predc.reshape(pred.shape + (3,))
    pred_dir = CONFIG.PRED_DIR + dirname + "/"
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    Image.fromarray(predc.astype('uint8'), 'RGB').save(pred_dir + "color_pred" + str(i) + ".png")


def scatter_error_rates(x1, y1, x2=None, y2=None, x3=None, y3=None):
    if x2 is None:
        x2 = []
    if y2 is None:
        y2 = []
    s = 10
    lw = 0.5
    fig, ax = plt.subplots(figsize=(6,3))
    plt.plot(x1, y1, "--", lw = lw, zorder=1)
    # plt.scatter(x1[0], y1[0], color="C2", zorder=4)
    plt.scatter(x1[1:], y1[1:], s = s, label="Bayes ML Interpol.", zorder=4)
    plt.plot([x1[0]] + x2,[y1[0]] + y2, "--", lw = lw, color="C2", alpha=0.5, zorder=1)
    plt.scatter([x1[0]] + x2, [y1[0]] + y2, s = s, c="C2", label="ML+MetaSeg+GB", zorder=3)
    if x3 is not None and y3 is not None:
        for i in range(len(x3)):
            if i == 0:
                plt.plot(x3[i] + [x1[-1]], y3[i] + [y1[-1]], "--", lw=lw, color="C1", alpha=1, zorder=1)
                plt.scatter(x3[i], y3[i], s=s/4, color="C1", label="Classif. Thresh.", alpha=1, zorder=2)
            else:
                plt.plot(x3[i] + [x1[-1]], y3[i] + + [y1[-1]], "--", lw=lw, color="C1", alpha=1, zorder=1)
                plt.scatter(x3[i], y3[i], s=s/4, color="C1", alpha=1, zorder=2)

    plt.xlabel("\# false positive segments")
    plt.ylabel("\# false negative segments")
    plt.legend()
    ax.annotate('Bayes', xy=(x1[0], y1[0]), xytext=(x1[0] + 50, y1[0]  ), fontsize=8)
    ax.annotate('ML', xy=(x1[-1], y1[-1]), xytext=(x1[-1] - 30, y1[-1] + 10), fontsize=8)
    ax.annotate('Fusion', xy=(x2[-1], y2[-1]),  xytext=(x2[-1]-400, y2[-1]-10), fontsize=8)
    if not os.path.exists(CONFIG.RESULTS_DIR):
        os.makedirs(CONFIG.RESULTS_DIR)
    plt.savefig(CONFIG.RESULTS_DIR + "scatter_error_rates.pdf", bbox_inches='tight', transparent=True)


def feature_importance_plot(mean, std, X_names, top=10):
    n = len(X_names)
    for i in range(n):
        name = "\_".join(X_names[i].split("_"))
        X_names[i] = name

    mean, std, X_names = zip(*sorted(zip(mean, std, X_names), reverse=True))
    mean = np.array(mean)[:top]
    X_names = np.array(X_names)[:top]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.set_axisbelow(True)
    ax.grid(color="silver", linestyle='--', alpha=0.3, linewidth=0.4)
    # ax.barh(np.arange(n), mean, xerr=std, align='center', ecolor='black', error_kw={"capsize": 3})
    ax.barh(np.arange(len(mean)), mean, align='center', ecolor='black', error_kw={"capsize": 3})
    ax.set_yticks(np.arange(len(mean)))
    ax.set_yticklabels(X_names)
    ax.invert_yaxis()  # X_names read top-to-bottom
    ax.set_xlabel('Feature Importance Score')
    # ax.set_xlabel('Importance score averaged over all cross validation splits')
    # ax.set_title('Gradient Boosting Feature Importance')

    fig.savefig(CONFIG.RESULTS_DIR + "z_feature_importance.pdf", bbox_inches='tight', transparent=True)
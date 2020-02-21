#!/usr/bin/env python3
"""
script including
functions that do calculations
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from global_defs import CONFIG


def regression_fit_and_predict(X_train, y_train, X_test):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_test_pred = np.clip(model.predict(X_test), 0, 1)
    y_train_pred = np.clip(model.predict(X_train), 0, 1)
    return y_test_pred, y_train_pred


def classification_l1_fit_and_predict(X_train, y_train, lambdas, X_test):
    if CONFIG.META_MODEL == "linear":
        model = linear_model.LogisticRegression(C=lambdas, penalty='l1', solver='saga', max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    y_test_pred = model.predict_proba(X_test)
    y_train_pred = model.predict_proba(X_train)
    return y_test_pred, y_train_pred, np.asarray(model.coef_[0])


def classification_fit_and_predict(X_train, y_train, X_test, balance=True):
    if CONFIG.META_MODEL == "linear":
        model = linear_model.LogisticRegression(solver='saga', max_iter=1000, tol=1e-3)
    elif CONFIG.META_MODEL == "gradientboosting":
        model = GradientBoostingClassifier(loss='exponential', n_estimators=27, max_depth=3, learning_rate=0.14, max_features=5)
    if balance:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    model.fit(X_train, y_train)
    y_test_pred = model.predict_proba(X_test)
    y_train_pred = model.predict_proba(X_train)
    return y_test_pred, y_train_pred, model.feature_importances_


def classification_fit_and_predict_cross_val(X, y, runs=10, splits=5, seed=123):
    np.random.seed(seed)
    thresholds = [thr / splits for thr in range(1, splits + 1)]
    tp_i = np.array([i for i in range(len(y)) if y[i] == 0])
    fp_i = np.array([i for i in range(len(y)) if y[i] != 0])
    y_proba = np.zeros((runs, len(y), 2))
    feat_importance = []
    for run in range(runs):
        rtp = np.random.uniform(size=np.sum(y == 0))
        rfp = np.random.uniform(size=np.sum(y == 1))
        for split, thr in enumerate(thresholds, 1):
            tp_i_train = tp_i[~np.logical_and(thr - 1 / float(splits) < rtp, rtp < thr)]
            fp_i_train = fp_i[~np.logical_and(thr - 1 / float(splits) < rfp, rfp < thr)]
            tp_i_test = tp_i[np.logical_and(thr - 1 / float(splits) < rtp, rtp < thr)]
            fp_i_test = fp_i[np.logical_and(thr - 1 / float(splits) < rfp, rfp < thr)]
            i_train = np.concatenate((tp_i_train, fp_i_train))
            i_test = np.concatenate((tp_i_test, fp_i_test))
            y_test_proba, _, importance_score = classification_fit_and_predict(X[i_train], y[i_train], X[i_test])
            feat_importance.append(importance_score)
            fpr, tpr, _ = roc_curve(y[i_test], y_test_proba[:, 1])
            print("split %d auc: %f" % (split, auc(fpr, tpr)))
            for ix, i in enumerate(i_test):
                y_proba[run, i] = y_test_proba[ix]
    scores = np.array(feat_importance)
    return np.mean(y_proba, axis=0), np.mean(scores, axis=0), np.std(scores, axis=0)


def compute_correlations(metrics):
    pd.options.display.float_format = '{:,.5f}'.format
    df_full = pd.DataFrame(data=metrics)
    df_full = df_full.copy().drop(["class", "iou0"], axis=1)
    features = df_full.copy().drop(["iou"], axis=1).columns
    df_all = df_full.copy()
    df_full = df_full.copy().loc[df_full['S_in'].nonzero()[0]]
    return df_all, df_full


def compute_metrics_from_heatmap(heatmap, components, comp_id):
    n_in = np.count_nonzero(components == comp_id)
    n_bd = np.count_nonzero(components == -comp_id)
    value = np.sum(heatmap[abs(components) == comp_id]) / (n_in + n_bd)
    value_in = np.sum(heatmap[components == comp_id]) / n_in if n_in > 0 else 0
    value_bd = np.sum(heatmap[components == -comp_id]) / n_bd
    value_rel = value * (n_in + n_bd) / n_bd
    value_rel_in = value_in * n_in / n_bd
    return [value, value_in, value_bd, value_rel, value_rel_in]


def compute_error_rate(metrics, class_filter=[]):
    if not class_filter:
        class_filter = np.unique(metrics["class"])
    errors = len([i for i in range(len(metrics["S"]))
                  if metrics["iou"][i] == 0
                  and metrics["class"][i] in class_filter
                  and metrics["S_in"][i] != 0])
    return errors


def compute_segment_errors_from_masks(pred, gt, comp_pred, comp_gt, label):
    fp, fn = (0, 0)
    comp_pred[pred != label] = 0
    comp_gt[gt != label] = 0
    for c in np.unique(comp_pred[comp_pred > 0]):
        if ~np.isin(label, gt[comp_pred == c]):
            fp += 1
    for c in np.unique(comp_gt[comp_gt > 0]):
        if ~np.isin(label, pred[comp_gt == c]):
            fn += 1
    return fp, fn

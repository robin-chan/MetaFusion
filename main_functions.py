#!/usr/bin/env python3
"""
script including
class objects called in main
"""

import numpy as np
import time
import os
import pickle
from multiprocessing import Pool
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc, accuracy_score, confusion_matrix

from global_defs import CONFIG
from metrics import compute_metrics_components, compute_metrics_mask, prediction
from helper import concatenate_metrics, metrics_to_dataset, get_lambdas, metrics_to_nparray
from in_out import get_save_path_input_i, get_save_path_prediction_i, probs_gt_load, metrics_dump, \
    components_dump, stats_dump, metrics_load, components_load, load_priors, prediction_dump, prediction_load
from plot import visualize_regression_prediction_i, plot_roc_curve, plot_regression, plot_classif, \
    plot_scatter, plot_classif_hist, save_prediction_mask, scatter_error_rates, feature_importance_plot
from calculate import regression_fit_and_predict, classification_l1_fit_and_predict, \
    classification_fit_and_predict_cross_val, compute_correlations, compute_metrics_from_heatmap, \
    compute_error_rate, compute_segment_errors_from_masks
import labels as labels


# ----------------------------#
class compute_metrics(object):
    # ----------------------------#

    def __init__(self, num_cores=1, num_imgs=len(os.listdir(CONFIG.INPUT_DIR)), rewrite=True):
        """
        object initialization
        :param num_cores: (int) number of cores used for parallelization
        :param num_imgs:  (int) number of images to be processed
        :param rewrite:   (boolean) overwrite existing files if True
        """
        self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        self.rewrite = rewrite
        if num_imgs == 0: num_imgs = len(os.listdir(CONFIG.INPUT_DIR))  # reinitialization if 0
        self.num_imgs = num_imgs if not hasattr(CONFIG, 'NUM_IMAGES') else CONFIG.NUM_IMAGES
        self.ml_priors = load_priors()
        self.label_person = labels.cs_name2trainId["person"].trainId
        self.label_rider = labels.cs_name2trainId["rider"].trainId

    def compute_metrics_per_image(self, rule="bayes", ground_truth_analysis=True, alpha=1.0):
        """
        perform metrics computation
        """
        print("calculating statistics")
        print("decision rule:", rule)
        print("alpha:", alpha)
        p_args = [(k, rule, ground_truth_analysis, alpha) for k in range(self.num_imgs)]
        Pool(self.num_cores).starmap(self.compute_metrics_i, p_args)

    def compute_metrics_i(self, i, rule, ground_truth_analysis, alpha):
        """
        perform metrics computation for one image
        :param i: (int) id of the image to be processed
        :param rule: (str) decision rule
        :param ground_truth_analysis: (boolean) if True, compute metrics for gt mask
        :param alpha: (float) degree of interpolation between ML and Bayes
        """
        if os.path.isfile(get_save_path_input_i(i)) and self.rewrite:
            start = time.time()
            probs, gt, _ = probs_gt_load(i)

            # map rider class to person by excluding rider class
            probs[:,:,self.label_rider] = 0
            probs /= np.sum(probs, axis=-1, keepdims=True)
            gt[gt == self.label_rider] = self.label_person
            pred_bay = prediction(probs, gt)
            if "bayes" in rule:
                metrics, components = compute_metrics_components(probs, gt)
                metrics_dump(metrics, i, "bayes")
                components_dump(components, i, "bayes")
                if ground_truth_analysis:
                    metrics_gt_bay, components_gt_bay = compute_metrics_mask(gt, pred_bay)
                    metrics_dump(metrics_gt_bay, i, "gt_bayes")
                    components_dump(components_gt_bay, i, "gt_bayes")
            if "ml" in rule:
                priors = (1 - alpha) * np.ones(self.ml_priors.shape, dtype="uint8") + alpha * self.ml_priors
                probs_ml = probs / priors
                probs_ml /= np.sum(probs_ml, axis=-1, keepdims=True)

                # map rider class to person by excluding rider class
                probs_ml[:,:,self.label_rider] = 0
                probs_ml /= np.sum(probs_ml, axis=-1, keepdims=True)

                pred_ml = prediction(probs_ml, gt)
                metrics_ml, components_ml = compute_metrics_components(probs_ml, gt)
                metrics_dump(metrics_ml, i, "ml_" + str(alpha))
                components_dump(components_ml, i, "ml_" + str(alpha))
                if alpha != 0:
                    metrics_bay_ml, _ = compute_metrics_mask(pred_ml, pred_bay)
                    metrics_dump(metrics_bay_ml, i, "ml_" + str(alpha) + "_bayes")
                if ground_truth_analysis:
                    metrics_gt_ml, components_gt_ml = compute_metrics_mask(gt, pred_ml)
                    metrics_dump(metrics_gt_ml, i, "gt_ml_" + str(alpha))
                    components_dump(components_gt_ml, i, "gt")
            print("image", i, "processed in {}s\r".format(round(time.time() - start)))

    def add_heatmaps_as_metric(self, heat_dir, key):
        """
        add another dispersion heatmap as metric/input for meta model
        :param heat_dir:  (str) directory with heatmaps as numpy arrays
        :param key:       (str) new key to access added metric
        """
        print("Add " + key + "  to metrics")
        p_args = [(heat_dir, key, k) for k in range(self.num_imgs)]
        Pool(self.num_cores).starmap(self.add_heatmap_as_metric_i, p_args)

    def add_heatmap_as_metric_i(self, heat_dir, key, i):
        """
        derive aggregated metrics per image and add to metrics dictionary
        :param heat_dir:  (str) directory with heatmaps as numpy arrays
        :param key:       (str) new key to access added metric
        :param i:         (int) id of the image to be processed
        """
        _, _, path = probs_gt_load(i)
        heat_name = os.path.basename(path)[:-4] + ".npy"
        heatmap = np.load(heat_dir + heat_name)
        metrics = metrics_load(i)
        components = components_load(i)
        keys = [key, key + "_in", key + "_bd", key + "_rel", key + "_rel_in"]
        heat_metric = {k: [] for k in keys}
        for comp_id in range(1, abs(np.min(components)) + 1):
            values = compute_metrics_from_heatmap(heatmap, components, comp_id)
            for j, k in enumerate(keys):
                heat_metric[k].append(values[j])
        metrics.update(heat_metric)
        metrics_dump(metrics, i)


# --------------------------------------#
class visualize_meta_prediction(object):
# --------------------------------------#

    def __init__(self, num_cores=1, num_imgs=len(os.listdir(CONFIG.INPUT_DIR))):
        """
        object initialization
        :param num_cores: (int) number of cores used for parallelization
        :param num_imgs:  (int) number of images to be processed
        """
        self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        if num_imgs == 0: num_imgs = len(os.listdir(CONFIG.INPUT_DIR))  # reinitialization if 0
        self.num_imgs = num_imgs if not hasattr(CONFIG, 'NUM_IMAGES') else CONFIG.NUM_IMAGES

    def visualize_regression_per_image(self):
        """
        perform metrics visualization
        """
        print("visualization running")

        metrics, start = concatenate_metrics(self.num_imgs, save=False)
        nclasses = np.max(metrics["class"]) + 1

        Xa, classes, ya, _, X_names, class_names = metrics_to_dataset(metrics, nclasses, non_empty=False)
        Xa = np.concatenate((Xa, classes), axis=-1)
        X_names += class_names

        ya_pred, _ = regression_fit_and_predict(Xa, ya, Xa)
        print("model r2 score:", r2_score(ya, ya_pred))
        print(" ")

        p_args = [(ya[start[i]:start[i + 1]], ya_pred[start[i]:start[i + 1]], i) for i in range(self.num_imgs)]
        Pool(self.num_cores).starmap(visualize_regression_prediction_i, p_args)


# ----------------------------#
class fusion_ml_metaseg(object):
# ----------------------------#

    def __init__(self, alpha=1.0, label=labels.cs_name2trainId["person"].trainId,
                 thresh=(0.5,), num_cores=1, num_imgs=len(os.listdir(CONFIG.INPUT_DIR))):
        """
        object initialization
        :param alpha:     (float) degree of interpolation between Bayes and ML
        :param label      (int) label id of class to be analyzed
        :param num_cores: (int) number of cores used for parallelization
        :param num_imgs:  (int) number of images to be processed
        """
        if thresh is None:
            thresh = [0.5]
        self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        self.num_imgs = num_imgs if not hasattr(CONFIG, 'NUM_IMAGES') else CONFIG.NUM_IMAGES
        self.alpha = alpha
        self.label = label
        self.ml_priors = load_priors()
        self.metrics_ml, self.start_ml = concatenate_metrics("ml_" + str(self.alpha), self.num_imgs)
        self.label_person = labels.cs_name2trainId["person"].trainId
        self.label_rider = labels.cs_name2trainId["rider"].trainId
        self.thresh = thresh

    def fusion(self, visualize=False, classification_thresh=False):
        """
        perform fusion of ML and MetaSeg
        """
        print("fusion ml & metaseg")
        print("alpha:", self.alpha)
        nclasses = np.max(self.metrics_ml["class"]) + 1

        disjoint_I = self.identify_disjoint_segments()
        # empty_I = self.identify_empty_segments()
        metrics = dict({})
        for m in self.metrics_ml:
            metrics[m] = [self.metrics_ml[m][i] for i in disjoint_I]

        Xa, classes, _, y0a, X_names, _ = metrics_to_dataset(metrics, nclasses)
        # Xa = np.concatenate((Xa, classes), axis=-1)

        y_probs, feat_importance_scores, feat_importance_std = classification_fit_and_predict_cross_val(Xa, y0a)
        feature_importance_plot(feat_importance_scores, feat_importance_std, X_names)
        print(len(X_names))
        exit()

        if not classification_thresh:
            self.thresh = (0.5,)

        for t in self.thresh:
            y_pred = (y_probs[:, 1] > t).astype("int")
            tn, fp, fn, tp = confusion_matrix(y0a, y_pred).ravel()
            print("TP:", tp, "FP:", fp, "FN:",fn, "TN:", tn)

            # keep all ML segments of class defined by 'self.label'
            # that are non-empty and have not been kicked by MetaSeg
            kick_I = np.unique([disjoint_I[i] for i in range(len(disjoint_I)) if y_pred[i] == 1]
                               + self.identify_empty_segments())
            keep_I = [i for i in self.identify_class_segments() if i not in kick_I]

            self.fuse_masks(keep_I, t, visualize)


    def fuse_masks(self, keep_I, thresh, visualize):
        """
        generate fused masks after processed by ML and MetaSeg
        :param keep_I: (list) segments IDs of ML segments to keep
        :param thresh: (str)
        :param visualize: (boolean)
        """
        t_string = "" if thresh == 0.5 else "_" + str(thresh)
        p_args = [(k, keep_I, t_string, visualize) for k in range(self.num_imgs)]
        Pool(self.num_cores if self.num_cores < 5 else 4).starmap(self.fuse_mask_i, p_args)


    def fuse_mask_i(self, i, keep_I, t_string, visualize):
        """
        generate fused mask of one image after processed by ML and MetaSeg
        :param i: (int) image id
        :param keep_I: (list)
        :param thresh: (str)
        :param visualize: (boolean)
        """
        start = time.time()
        keep_I = [j+1 - self.start_ml[i] for j in keep_I if self.start_ml[i] <= j < self.start_ml[i + 1]]
        probs, gt, _ = probs_gt_load(i)

        # map rider class to person by excluding rider class
        probs[:, :, self.label_rider] = 0
        probs /= np.sum(probs, axis=-1, keepdims=True)
        gt[gt == self.label_rider] = self.label_person

        priors = (1 - self.alpha) * np.ones(self.ml_priors.shape, dtype="uint8") + self.alpha * self.ml_priors
        pred_bay = prediction(probs, gt)
        pred_ml = prediction(probs / priors, gt)
        # components_bay = np.absolute(components_load(i, "ml_0.0"))
        components_ml = np.absolute(components_load(i, "ml_" + str(self.alpha)))

        fusion = np.copy(pred_bay)
        fusion[np.isin(components_ml, keep_I)] = pred_ml[np.isin(components_ml, keep_I)]

        # components_fusion = components_bay
        # components_fusion[np.isin(components_ml, keep_I)] = components_ml[np.isin(components_ml, keep_I)]

        if not os.path.exists(os.path.dirname(get_save_path_prediction_i(i, "ml_0.0"))):
            prediction_dump(pred_bay, i, "ml_0.0")
        prediction_dump(pred_ml, i, "ml_" + str(self.alpha))
        prediction_dump(fusion, i, "fusion_" + str(self.alpha) + t_string)
        # components_dump(components_fusion, i, "fusion_" + str(self.alpha))

        if visualize:
            top = np.concatenate((pred_bay, pred_ml), axis=1)
            bot = np.concatenate((gt, fusion), axis=1)
            collage = np.concatenate((top, bot), axis=0)
            save_prediction_mask(collage, i, "fusion_" + str(self.alpha) + t_string + "/collage")

        print("image", i, "processed in {}s\r".format(round(time.time() - start)))

    def identify_disjoint_segments(self):
        """
        identify non-empty disjoint Bayes and ML segments
        :return: list of segment Ids
        """
        metrics, start = concatenate_metrics("ml_" + str(self.alpha) + "_bayes", self.num_imgs)
        disjoint_ids = [i for i in range(len(metrics["S"]))
                        if metrics["class"][i] == self.label
                        and metrics["iou"][i] == 0
                        and metrics["S_in"][i] > 0]
        return disjoint_ids

    def identify_class_segments(self):
        """
        identify empty ML segments belonging to specified class
        :return: list of segment Ids
        """
        segment_ids = [i for i in range(len(self.metrics_ml["S"])) if self.metrics_ml["class"][i] == self.label]
        return segment_ids

    def identify_empty_segments(self):
        """
        identify empty ML segments
        :return: list of segment Ids
        """
        empty_ids = [i for i in range(len(self.metrics_ml["S"]))
                     if self.metrics_ml["class"][i] == self.label
                     and self.metrics_ml["S_in"][i] == 0]
        return empty_ids

# ----------------------------#
class analyze_fusion(object):
# ----------------------------#

    def __init__(self, label=labels.cs_name2trainId["person"].trainId, thresh=(0.5,),
                 num_cores=1, num_imgs=len(os.listdir(CONFIG.INPUT_DIR)), alphas=None, rewrite=True):
        """
        object initialization
        :param label      (int) label id of class to be analyzed
        :param num_cores: (int) number of cores used for parallelization
        :param num_imgs:  (int) number of images to be processed
        """
        self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        if num_imgs == 0: num_imgs = len(os.listdir(CONFIG.INPUT_DIR))  # reinitialization if 0
        self.num_imgs = num_imgs if not hasattr(CONFIG, 'NUM_IMAGES') else CONFIG.NUM_IMAGES
        self.rewrite = rewrite
        self.label = label
        self.thresh = thresh
        if alphas is None:
            self.alphas = [x[len("fusion_"):] for x in os.listdir(CONFIG.PRED_DIR) if x.startswith("fusion_")]
        else:
            self.alphas = [str(i) for i in alphas]
        self.label_person = labels.cs_name2trainId["person"].trainId
        self.label_rider = labels.cs_name2trainId["rider"].trainId


    def compute_metrics_per_image(self):
        """
        perform metrics computation
        """
        print("calculating errors...")
        for alpha in self.alphas:
            for t in self.thresh:
                t_string = "" if t == 0.5 else "_" + str(t)
                print("alpha:", alpha)
                print("classification threshold:", t)
                p_args = [(k, alpha, t_string) for k in range(self.num_imgs)]
                Pool(self.num_cores).starmap(self.compute_metrics_i, p_args)


    def compute_metrics_i(self, i, alpha, thresh):
        if os.path.isfile(get_save_path_input_i(i)) and self.rewrite:
            start = time.time()
            _, gt, _ = probs_gt_load(i)
            gt[gt == self.label_rider] = self.label_person
            pred_fusion = prediction_load(i, "fusion_" + alpha + thresh)
            ml_fusion, _ = compute_metrics_mask(pred_fusion, gt)
            gt_fusion, _ = compute_metrics_mask(gt, pred_fusion)
            metrics_dump(ml_fusion, i, "ml_fusion_" + alpha + thresh)
            metrics_dump(gt_fusion, i, "gt_fusion_" + alpha + thresh)
            print("image", i, "processed in {}s\r".format(round(time.time() - start)))


    def compute_errors_from_dict(self):
        x1, x2, y1, y2, x3, y3 = [], [], [], [], [], []
        x1.append(len(self.identify_empty_intersection("ml_0.0")))
        y1.append(len(self.identify_empty_intersection("gt_ml_0.0")))
        for alpha in self.alphas:
            print("alpha:", alpha)

            x1.append(len(self.identify_empty_intersection("ml_" + str(alpha))))
            y1.append(len(self.identify_empty_intersection("gt_ml_" + str(alpha))))
            x2.append(len(self.identify_empty_intersection("ml_fusion_" + str(alpha))))
            y2.append(len(self.identify_empty_intersection("gt_fusion_" + str(alpha))))

            print("False-positives:")
            print("{:<8}".format("Bayes:"), x1[0])
            print("{:<8}".format("ML:"), x1[-1])
            print("{:<8}".format("Fusion:"), x2[-1])

            print("False-negatives:")
            print("{:<8}".format("Bayes:"), y1[0])
            print("{:<8}".format("ML:"), y1[-1])
            print("{:<8}".format("Fusion:"), y2[-1])

            if alpha == self.alphas[-1]:
                x3_t, y3_t = [], []
                for t in self.thresh:
                    t_string = "" if t == 0.5 else "_" + str(t)
                    x3_t.append(len(self.identify_empty_intersection("ml_fusion_" + str(alpha)+ t_string)))
                    y3_t.append(len(self.identify_empty_intersection("gt_fusion_" + str(alpha) + t_string)))
                x3.append(x3_t)
                y3.append(y3_t)

        return x1, x2, y1, y2, x3, y3


    def scatter_error(self):
        print("plot scatter")
        x1, x2, y1, y2, x3, y3 = self.compute_errors_from_dict()
        scatter_error_rates(x1, y1, x2, y2, [], [])


    def identify_empty_intersection(self, rule):
        metrics, _ = concatenate_metrics(rule, self.num_imgs)
        if "gt" in rule:
            ids = [i for i in range(len(metrics["S"]))
                   if metrics["iou"][i] == 0
                   and metrics["class"][i] == self.label]
        else:
            ids = [i for i in range(len(metrics["S"]))
                   if metrics["iou"][i] == 0
                   and metrics["S_in"][i] > 0
                   and metrics["class"][i] == self.label]
        return ids
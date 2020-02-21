#!/usr/bin/env python3
"""
script including
class object with global settings
"""


class CONFIG:

    # ---------------------#
    # set necessary paths #
    # ---------------------#

    metaseg_io_path = "/home/chan/metafusion_io/"

    # ----------------------------#
    # paths for data preparation #
    # ----------------------------#

    # IMG_DIR   = "/home/chan/datasets/cityscapes/leftImg8bit/val/"
    # GT_DIR    = "/home/chan/datasets/cityscapes/gtFine/"
    # PROBS_DIR = "/home/chan/datasets/cityscapes/npProbs/val/"

    # ------------------#
    # select or define #
    # ------------------#

    datasets    = ["cityscapes"]
    model_names = ["xc.mscl.os8", "mn.sscl.os16"]
    meta_models = ["linear", "neural", "gradientboosting"]

    DATASET     = datasets[0]
    MODEL_NAME  = model_names[1]
    META_MODEL  = meta_models[2]

    # PROBS_DIR = "/home/chan/datasets/cityscapes/npProbs/val/" + MODEL_NAME

    # --------------------------------------------------------------------#
    # select tasks to be executed by setting boolean variable True/False #
    # --------------------------------------------------------------------#

    COMPUTE_METRICS     = False
    VISUALIZE_RATING    = False
    ANALYZE_METRICS     = False
    FUSION              = True
    ANALYZE_FUSION      = False

    # -----------#
    # optionals #
    # -----------#

    NUM_CORES = 40
    # NUM_IMAGES = 10

    INPUT_DIR       = metaseg_io_path + "input/"        + DATASET + "/" + MODEL_NAME + "/"
    METRICS_DIR     = metaseg_io_path + "metrics/"      + DATASET + "/" + MODEL_NAME + "/"
    COMPONENTS_DIR  = metaseg_io_path + "components/"   + DATASET + "/" + MODEL_NAME + "/"
    IOU_SEG_VIS_DIR = metaseg_io_path + "iou_seg_vis/"  + DATASET + "/" + MODEL_NAME + "/"
    RESULTS_DIR     = metaseg_io_path + "results/"      + DATASET + "/" + MODEL_NAME + "/"
    STATS_DIR       = metaseg_io_path + "stats/"        + DATASET + "/" + MODEL_NAME + "/"
    PRED_DIR        = metaseg_io_path + "predictions/"  + DATASET + "/" + MODEL_NAME + "/"
    PRIORS_DIR      = metaseg_io_path + "priors/"       + DATASET + "/"

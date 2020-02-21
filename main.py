#!/usr/bin/env python3
"""
main script executing tasks defined in global settings file
"""

from global_defs import CONFIG
from prepare_data import prepare_data
from main_functions import compute_metrics, visualize_meta_prediction, fusion_ml_metaseg, analyze_fusion


alphas = [0.9, 0.95, 0.975, 0.99, 0.995, 1.0]
# alphas = [0.95, 0.99, 0.995, 1.0]
# alphas = [1.0]

thresh = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
# thresh = [0.0, 1.0]

def main():
    if CONFIG.COMPUTE_METRICS:
        run = compute_metrics()
        # run.compute_metrics_per_image("gt_bayes")
        # for alpha in [0.0] + alphas:
        for alpha in alphas:
            run.compute_metrics_per_image(rule="ml", ground_truth_analysis=True, alpha=alpha)

    if CONFIG.VISUALIZE_RATING:
        run = visualize_meta_prediction()
        run.visualize_regression_per_image()

    if CONFIG.FUSION:
        for alpha in alphas:
            run = fusion_ml_metaseg(alpha=alpha, thresh=thresh)
            run.fusion()
            # run.fuse_masks()

    if CONFIG.ANALYZE_FUSION:
        run = analyze_fusion(alphas=alphas, thresh=thresh)
        # run.compute_metrics_per_image()
        run.scatter_error()
        # run.compute_errors_from_dict()
        # run.compute_errors()

if __name__ == '__main__':
    print("===== METASEG START =====")
    main()
    print("===== METASEG DONE! =====")

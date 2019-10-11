import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend_handler import HandlerLine2D


def report_best_cv_scores(results):
    best_model = np.flatnonzero(results["rank_test_score"] == 1)[0]
    print("Best model scores")
    print(
        "Mean validation score: {:.3f} (std: {:.3f})".format(
            results["mean_test_score"][best_model],
            results["std_test_score"][best_model],
        )
    )
    print(
        "Mean train score: {:.3f} (std: {:.3f})".format(
            results["mean_train_score"][best_model],
            results["std_train_score"][best_model],
        )
    )


def plot_feature_importances(model, df):
    feat_importances = pd.Series(model.feature_importances_, index=df.columns)
    feat_importances.sort_values().plot(kind="barh")
    plt.xlabel("Relative Importance")
    sns.despine()


def plot_max_depth_train_val_scores(train_scores, validation_scores, max_depths):
    line1, = plt.plot(max_depths, train_scores, "b", label="Train R2")
    line2, = plt.plot(max_depths, validation_scores, "r", label="Validation R2")

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.xticks(np.arange(min(max_depths), max(max_depths) + 1, 4.0))
    plt.ylabel("R2 score")
    plt.xlabel("Tree depth")
    plt.grid(True)
    sns.despine()


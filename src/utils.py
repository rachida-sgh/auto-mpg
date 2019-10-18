import os
import matplotlib.pyplot as plt


def data_path(*path):
    this_dir = os.path.realpath(os.path.dirname(__file__))
    return os.path.join(this_dir, "..", "data", *path)


def save_fig(fig_path, fig_id, fig_extension="png"):
    path = os.path.join(fig_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    plt.tight_layout()
    plt.savefig(path, format=fig_extension)


def split_features_target(df, target):
    return df.drop(target, axis=1), df[target]

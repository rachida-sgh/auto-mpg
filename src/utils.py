import os
import urllib
import matplotlib.pyplot as plt

ROOT_DIR = ".."


def data_path(*path):
    this_dir = os.path.realpath(os.path.dirname(__file__))
    return os.path.join(this_dir, "..", "data", *path)


def download_data(file_name="auto-mpg.data"):
    raw_data_dir = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    file_path = data_path("raw",file_name)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    urllib.request.urlretrieve(url, file_path)
    print("downloading data to", os.path.join(raw_data_dir, "auto-mpg.data"))


def save_fig(fig_path, fig_id, fig_extension="png"):
    path = os.path.join(fig_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    plt.tight_layout()
    plt.savefig(path, format=fig_extension)


def split_features_target(df, target):
    return df.drop(target, axis=1), df[target]

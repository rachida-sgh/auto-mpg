import os
import urllib
import matplotlib.pyplot as plt
import pandas as pd
import pickle

ROOT_DIR = ".."


def data_path(*path):
    this_dir = os.path.realpath(os.path.dirname(__file__))
    return os.path.join(this_dir, "..", "data", *path)


def download_data(file_name="auto-mpg.data"):
    raw_data_dir = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    file_path = data_path("raw", file_name)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    urllib.request.urlretrieve(url, file_path)
    print("downloading data to", os.path.join(raw_data_dir, "auto-mpg.data"))


def save_data(df, data_state, file_name):
    data_dir = os.path.join(ROOT_DIR, "data", data_state)
    os.makedirs(data_dir, exist_ok=True)
    df.to_pickle(os.path.join(data_dir, file_name))
    
def pickle_model(model, model_path):
    model_dir = os.path.join(ROOT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, model_path), "wb") as f:
        pickle.dump(model, f)
        
def load_model(model_path):
    full_model_path = os.path.join(ROOT_DIR, "model", model_path)
    with open(full_model_path, "rb") as file:
        model = pickle.load(file)
    return model


def save_fig(fig_path, fig_id, fig_extension="png"):
    path = os.path.join(fig_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    plt.tight_layout()
    plt.savefig(path, format=fig_extension)


def split_features_target(df, target):
    return df.drop(target, axis=1), df[target]

import os

def data_path(*path):
    this_dir = os.path.realpath(os.path.dirname(__file__))
    print(this_dir)
    return os.path.join(this_dir, "..", "data", *path)

def split_features_target(df, target):
    return df.drop(target, axis=1), df[target]
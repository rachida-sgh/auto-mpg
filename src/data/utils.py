import os


def data_path(*path):
    this_dir = os.path.realpath(os.path.dirname(__file__))
    return os.path.join(this_dir, "..", "..", "data", *path)

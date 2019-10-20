import pandas as pd
import numpy as np
import os
from src.utils import data_path


def load_raw_data(file_name="auto-mpg.data"):
    file_path = data_path("raw", file_name)
    return pd.read_csv(
        file_path,
        delim_whitespace=True,
        header=None,
        names=[
            "mpg",
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "year",
            "origin",
            "name",
        ],
    )


def correct_company_names(df):
    typos = {
        "Chevroelt": "Chevrolet",
        "Toyouta": "Toyota",
        "Vw": "Volkswagen",
        "Vokswagen": "Volkswagen",
        "Mercedes-Benz": "Mercedes",
        "Chevy": "Chevrolet",
        "Maxda": "Mazda",
        "Amc": "AMC",
        "Bmw": "BMW",
    }

    df["name"] = df["name"].str.title()
    df["company"] = df["name"].str.split(" ").str[0]

    for typo in typos:
        df["company"] = df["company"].str.replace(typo, typos[typo])


def get_region_names(df):
    region_map = {1: "USA", 2: "EUROPE", 3: "ASIA"}
    df["region"] = df["origin"].map(region_map)
    df.drop("origin", axis=1, inplace=True)


def get_clean_dataset():
    df = load_raw_data("auto-mpg.data")
    # convert horsepower column (object) it to int
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    correct_company_names(df)
    get_region_names(df)
    df.to_pickle(data_path("interim", "data_cleaned.pkl"))
    return df

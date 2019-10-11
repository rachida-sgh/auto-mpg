import pandas as pd
import numpy as np
import os

from src.data.utils import data_path


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Continous features
CONTINUOUS_FEATURES = ["displacement", "horsepower", "weight", "acceleration"]
# Categorical features
ORDINAL_FEATURES = ["cylinders", "year"]
NOMINAL_FEATURES = ["region"]


def make_final_transformation_pipe():

    # Build transformation pipelines adapted to feature types
    cont_pipeline = Pipeline(
        [
            ("imputer_cont", SimpleImputer(strategy="median")),
            ("std_scaler_cont", StandardScaler()),
        ]
    )

    ord_pipeline = Pipeline(
        [
            ("imputer_ord", SimpleImputer(strategy="most_frequent")),
            ("std_scaler_ord", StandardScaler()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ("cont", cont_pipeline, CONTINUOUS_FEATURES),
            ("ord", ord_pipeline, ORDINAL_FEATURES),
            ("nom", OneHotEncoder(), NOMINAL_FEATURES),
        ]
    )

    return full_pipeline


def get_interim_data(dataset):
    if dataset not in ["train", "test"]:
        raise Exception("dataset type argument is train or test)")
    filename = f"df_{dataset}_cleaned.csv"
    filepath = data_path("interim", filename)
    return pd.read_csv(filepath)


def make_final_train_set():
    df_train = get_interim_data("train")
    X_train = df_train.drop("mpg", axis=1)
    y_train = df_train["mpg"]

    full_pipeline = make_final_transformation_pipe()
    X_train_processed_values = full_pipeline.fit_transform(X_train)

    # Add columns names to build the processed dataframe
    region_ohe_features = list(
        full_pipeline.named_transformers_["nom"].get_feature_names()
    )
    column_names = CONTINUOUS_FEATURES + ORDINAL_FEATURES + region_ohe_features
    X_train_processed = pd.DataFrame(X_train_processed_values, columns=column_names)

    # Drop one of the ohe features to limit correlations in the data set
    X_train_processed.drop("x0_EUROPE", axis=1, inplace=True)

    # Save the data
    df_train_processed = X_train_processed.join(y_train)
    df_train_processed.to_csv(data_path("processed", "df_train_processed.csv"))

    return column_names, full_pipeline, df_train_processed


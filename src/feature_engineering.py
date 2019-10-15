import pandas as pd
import numpy as np
import os

from src.utils import data_path, split_features_target


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
    filename = f"{dataset}_cleaned.pkl"
    filepath = data_path("interim", filename)
    return pd.read_pickle(filepath)


def make_final_sets():
    df_train = get_interim_data("train")
    df_test = get_interim_data("test")
    X_train, y_train = split_features_target(df_train, "mpg")
    X_test, y_test = split_features_target(df_test, "mpg")

    full_pipeline = make_final_transformation_pipe()
    X_train_processed_values = full_pipeline.fit_transform(X_train)
    X_test_processed_values = full_pipeline.transform(X_test)
    # Add columns names to build the processed dataframe
    region_ohe_features = list(
        full_pipeline.named_transformers_["nom"].get_feature_names()
    )
    column_names = CONTINUOUS_FEATURES + ORDINAL_FEATURES + region_ohe_features
    X_train_processed = pd.DataFrame(X_train_processed_values, columns=column_names)
    X_test_processed = pd.DataFrame(X_test_processed_values, columns=column_names)

    # Drop one of the ohe features to limit correlations in the data set
    for df in (X_train_processed, X_test_processed):
        df.drop("x0_EUROPE", axis=1, inplace=True)

    # Save the data
    df_train_processed = X_train_processed.join(y_train)
    df_train_processed.to_csv(data_path("processed", "train_processed.pkl"))

    df_test_processsed = X_test_processed.join(y_test)
    df_test_processsed.to_csv(data_path("processed", "test_processed.pkl"))

    return df_train_processed, df_test_processsed
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.utils import data_path

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


# def get_interim_data(dataset):
#     if dataset not in ["train", "test"]:
#         raise Exception("dataset type argument is train or test)")
#     filename = f"{dataset}_cleaned.pkl"
#     filepath = data_path("interim", filename)
#     return pd.read_pickle(filepath)


def get_cleaned_train_test_df():
    clean_data_path = data_path("interim", "data_cleaned.pkl")
    df = pd.read_pickle(clean_data_path)
    X = df.drop("mpg", axis=1)
    y = df["mpg"]
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


def make_final_sets():
    X_train, X_test, y_train, y_test = get_cleaned_train_test_df()

    full_pipeline = make_final_transformation_pipe()
    X_train_processed_values = full_pipeline.fit_transform(X_train)
    X_test_processed_values = full_pipeline.transform(X_test)

    # Add column names to build the processed dataframe
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
    df_train_processed = X_train_processed.join(y_train.reset_index(drop=True))
    df_train_processed.to_pickle(data_path("processed", "train_processed.pkl"))

    df_test_processsed = X_test_processed.join(y_test.reset_index(drop=True))
    df_test_processsed.to_pickle(data_path("processed", "test_processed.pkl"))

    return df_train_processed, df_test_processsed

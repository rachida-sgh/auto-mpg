import os
import pandas as pd
import numpy as np


def get_colmn_names(error):
    if error == "mse":
        error_colmn = "r" + error
        std_error_colmn = "r{}_std".format(error)
    else:
        error_colmn = error
        std_error_colmn = "{}_std".format(error)
    return error_colmn, std_error_colmn


def get_df_errors(cv_results):
    df_errors = pd.DataFrame(
        columns=["rmse", "rmse_std", "r2", "r2_std"], index=["train", "test"]
    )
    for set_ in ["train", "test"]:
        for error in ["mse", "r2"]:
            cv_error_name = "{}_{}".format(set_, error)
            error_colmn, std_error_colmn = get_colmn_names(error)
            if error == "mse":
                df_errors.loc[set_, error_colmn] = np.sqrt(
                    cv_results[cv_error_name]
                ).mean()
                df_errors.loc[set_, std_error_colmn] = np.sqrt(
                    cv_results[cv_error_name]
                ).std()
            else:
                df_errors.loc[set_, error_colmn] = cv_results[cv_error_name].mean()
                df_errors.loc[set_, std_error_colmn] = cv_results[cv_error_name].std()
    return df_errors.astype(float).round(3)


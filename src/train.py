import os
import pandas as pd
import numpy as np

###### Report function for model selection and training
def report_best_cv_scores(results):
    # get the rank of the best model based on test score
    best_model = np.flatnonzero(results["rank_test_score"] == 1)[0]
    print("Best model scores")
    print(
        "Mean validation RMSE: {:.3f} (std: {:.3f})".format(
            np.sqrt(-results["mean_test_score"][best_model]),
            np.sqrt(results["std_test_score"][best_model]),
        )
    )
    print(
        "Mean train RMSE: {:.3f} (std: {:.3f})".format(
            np.sqrt(-results["mean_train_score"][best_model]),
            np.sqrt(results["std_train_score"][best_model]),
        )
    )


###### Report function for algorithm selection
def rmse_cross_validate(cv_results):
    df_errors = pd.DataFrame(columns=["rmse", "rmse_std"], index=["train", "test"])
    for set_ in ["train", "test"]:
        error_key = "{}_mse".format(set_)
        rmse = np.sqrt(cv_results[error_key]).mean()
        rmse_std = np.sqrt(cv_results[error_key]).std()
        df_errors.loc[set_, "rmse"] = rmse
        df_errors.loc[set_, "rmse_std"] = rmse_std

    return df_errors.astype(float).round(3)

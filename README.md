Gas Mileage Prediction
============================== 

## Objective 

Given different features (or explanatory variables), the aim is to predict the fuel consumption in MPG (miles per gallon).

Such a model can be then used during the design process of cars case fuel consumption is a determinant factor to assess different designs and configurations. 

## Data Set

The data set comes from the UCI [Machine Learning Repository](https://archive.ics.uci.edu/ml/) and can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/auto+mpg).

The dataset is small (398 instances and 8 features). Yet, it is rich in terms of features types. 

- cylinders: numerical discrete
- displacement (engine size): continuous
- horsepower: conitnous
- weight: continuous
- acceleration: continuous
- model year: numerical discrete
- origin: numerical discrete
- car name: string (unique for each instance) 

 ## Approach Summary and Results 

After extensive data exploration and cleaning, we preprocessed original features and used them to train a Random Forest model.

We used Random Search Cross-Validation to tune the hyperparameters. The the weight of the car and the size of its engine are the most importante features to predict the fuel consumption accorfing the the Random Forest model.

The __RMSE__ (Root Mean Squared Error) score for the selected model is __2.27 MPG__ with a __95% confidence interval [1.82, 2.65]__.


Project Organization
------------

The project organisation is based on a minimalistic version of the [cookiecutter data science project template](https://cookiecutter.readthedocs.io/en/latest/installation.html). 

    ├── LICENSE
    ├── README.md          <- The top-level README.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Trained and serialized models
    ├── notebooks          <- Project walkthough in the form of sequenced Jupyter notebooks. 
    ├── figures           <- Generated graphics and figures to be used in reporting
    ├── requirements.txt   <- The requirements file
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    

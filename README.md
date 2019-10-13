Gas Mileage Prediction
============================== 
For this regression problem, the objective is to predict the fuel consumption in MPG (miles per gallon) using multiple features of an automobile. 

Such a model can be used during the design process of cars case fuel consumption is a determinant factor to assess different designs and configurations. 

## Data Set

The data set comes from the UCI [Machine Learning Repository](https://archive.ics.uci.edu/ml/) and can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/auto+mpg).

The dataset is small (398 instances and 8 features). Yet, it is rich in terms of features types. 

- cylinders: numerical discrete
- displacement: continuous
- horsepower: conitnous
- weight: continuous
- acceleration: continuous
- model year: numerical discrete
- origin: numerical discrete
- car name: string (unique for each instance) 

 ## Approach Summary and Results 

After extensive data exploration and cleaning, I preprocessed original features and used them to train a Random Forest model.

I used Random Search Cross-Validation to tune the hyperparameters. The weight of the car is the number one predictor for its fuel consumption.

The RMSE (Root Mean Squared Error) score for the selected model is 2.29 MPG with a 95% confidence interval [1.84, 2.67]. R2 score is 0.90.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

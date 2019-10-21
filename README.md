# Gas Mileage Prediction

- [Project Description](#project-description)
  - [Objectives](#objectives)
  - [Dataset](#dataset)
  - [Approach and results](#approach-and-results)
- [Running the project](#running-the-project)
  - [Installation](#installation)
  - [Running the Notebooks](#running-the-notebooks)
  - [Project Organization](#project-organization)


## Project Description

### Objectives

Given different features (or explanatory variables), the aim is to predict the fuel consumption in MPG (miles per gallon).

Such a model can be then used during the design process of cars case fuel consumption is a determinant factor to assess different designs and configurations.

### Dataset

The data set comes from the UCI [Machine Learning Repository](https://archive.ics.uci.edu/ml/) and can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/auto+mpg).

The dataset is small (398 instances and 8 features). Yet, it is rich in terms of features types.

- cylinders: numerical discrete
- displacement (engine size): continuous
- horsepower: continuous
- weight: continuous
- acceleration: continuous
- model year: numerical discrete
- origin: numerical discrete
- car name: string (unique for each instance)

### Approach and Results

After data exploration and cleaning, we preprocessed original features and used them to train a Random Forest model.

We used Random Search Cross-Validation to tune the hyperparameters. The weight of the car and the size of its engine are the most important features to predict the fuel consumption according to the Random Forest model.

The __RMSE__ (Root Mean Squared Error) score for the selected model is __2.27 MPG__ with a __95% confidence interval [1.82, 2.65]__.

The Jupyter notebooks will walk you through each step of the process. They are intended to be self-explanatory.

## Looking at the notebooks without executing any code

You can browse the notebooks [here](https://nbviewer.jupyter.org/github/rachida-sgh/auto-mpg/blob/master/notebooks/index.ipynb) using Jypyter nbviewer. Github works too but it can be slow. 

## Running the project

### Installation

To run the project you will need the following dependencies:

- `git`
- `miniconda` or `anaconda`

Clone the repository:

```bash
cd $HOME # or any directory you prefer`
git clone https://github.com/rachida-sgh/auto-mpg.git
cd auto-mpg
```

Create a conda environment with python 3.6:

```bash
conda create -n auto-mpg python=3.6
conda activate auto-mpg
```

This creates a new Python 3.6 environment called `auto-mpg`. You can change the name of the environment but it is recommended to have the same name for both the project and the virtual environment.

Once in the virtual environment, install the python dependencies and the project itself:

```bash
pip install -r requirements.txt
pip install -e .
```

### Running the notebooks

Start Jupyter:

```bash
jupyter notebook notebooks
```

You should see Jupyter's tree view of the notebooks in your browser. If your browser doesn't open automatically, visit http://localhost:8888 and click on `index.ipynb`.


### Project Organization

The project organisation is based on a minimalistic version of the [cookiecutter data science project template](https://cookiecutter.readthedocs.io/en/latest/installation.html).

    ├── LICENSE
    ├── README.md          <- The top-level README.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Trained and serialized models
    ├── notebooks          <- Project walkthough in the form of sequenced Jupyter notebooks.
    ├── figures            <- Generated graphics and figures to be used in reporting
    ├── requirements.txt   <- The requirements file
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.

import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Continous features
continuous_features = ['displacement',
                       'horsepower',
                       'weight',
                       'acceleration',]
# Categorical features
ordinal_features = ['cylinders',
                    'year',]    
nominal_features = ['region']   

def data_path(*path):
    this_dir = os.path.realpath(os.path.dirname(__file__))
    return os.path.join(this_dir, "..", "data", *path)

def load_raw_data(file_name):
    file_path = data_path("raw", file_name)
    return pd.read_csv(file_path,
                     delim_whitespace = True, header=None,
                     names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                              'year', 'origin', 'name'])

def correct_company_names(df):
    typos = {'Chevroelt' : 'Chevrolet',
            'Toyouta' : 'Toyota',
            'Vw' : 'Volkswagen',
            'Vokswagen' : 'Volkswagen',
            'Mercedes-Benz' : 'Mercedes',
            'Chevy' : 'Chevrolet',
            'Maxda' : 'Mazda',
            'Amc' : 'AMC',
            'Bmw' : 'BMW',}
    
    df['name'] = df['name'].str.title()
    df['company'] = df['name'].str.split(' ').str[0]
    
    for typo in typos:
        df['company'] = df['company'].str.replace(typo, typos[typo])

def get_region_names(df):
    region_map = {1 : 'USA',
                  2 : 'EUROPE',
                  3 : 'ASIA'}
    df['region'] = df['origin'].map(region_map)
    df.drop('origin', axis=1, inplace=True)
    
    
def get_clean_dataset():
    df = load_raw_data('auto-mpg.data')
    # convert horsepower column (object) it to int
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerc')
    correct_company_names(df)
    get_region_names(df)
    df.to_csv(data_path('interim', 'data_cleaned.csv'))
    return df

def make_final_transformation_pipe():

    # Build transformation pipelines adapted to feature types
    cont_pipeline = Pipeline([
        ('imputer_cont', SimpleImputer(strategy='median')),
        ('std_scaler_cont', StandardScaler()),
    ])  

    ord_pipeline = Pipeline([
        ('imputer_ord', SimpleImputer(strategy='most_frequent')),
        ('std_scaler_ord', StandardScaler()),
    ])  

    full_pipeline = ColumnTransformer([
        ('cont', cont_pipeline, continuous_features),
        ('ord', ord_pipeline, ordinal_features),
        ('nom', OneHotEncoder(), nominal_features),
    ])
    
    return full_pipeline

def get_interim_data(dataset):
    if dataset not in ['train', 'test']:
        raise Exception('dataset type argument is train or test)')
    filename = f'df_{dataset}_cleaned.csv'
    filepath = data_path('interim', filename)
    return pd.read_csv(filepath)
    

def make_final_train_set():
    df_train = get_interim_data('train')
    X_train = df_train.drop('mpg', axis=1)
    y_train = df_train['mpg']
    
    full_pipeline = make_final_transformation_pipe()
    X_train_processed_values = full_pipeline.fit_transform(X_train)

    # Add columns names to build the processed dataframe 
    region_ohe_features = list(full_pipeline.named_transformers_['nom'].get_feature_names())
    column_names = continuous_features + ordinal_features + region_ohe_features
    X_train_processed = pd.DataFrame(X_train_processed_values, columns=column_names)
    
    # Drop one of the ohe features to limit correlations in the data set
    X_train_processed.drop('x0_EUROPE', axis=1, inplace=True)
    
    # Save the data
    df_train_processed = X_train_processed.join(y_train)
    df_train_processed.to_csv(data_path("processed", "df_train_processed.csv"))
    
    return column_names, full_pipeline, df_train_processed
    
    
    
    
    


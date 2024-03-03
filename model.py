# MLflow
import mlflow
from mlflow.models import infer_signature

# for plotting
import io

# MLflow model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Visualizations
import matplotlib.pyplot as plt

# math functions
import numpy as np

# read CSV file
import os
import glob

# data frames
import pandas as pd

# regexes
import re

def data_analysis(df):
    # US-only, because keywords are in English
    df = df[df['country'] == 'US']
    df = df[df['category'].str.contains('"parent_name":"Technology"')]

    # List of regular expressions for search terms
    search_terms = []
    search_acronyms = ['ai','ml','nlp']
    search_words = ['artificial intelligence', 'neural networks', 'computer vision', 'natural language processing', 'deep learning',
                    'Tensorflow','robotics', 'chatbot', 'Augmented Reality', 'Speech Generation']
    for term in search_acronyms:
        pattern = r'[ (]' + ''.join([f'{char}[.]?' for char in term]) + r'[ ,.)]'
        search_terms.append(pattern)
    for term in search_words:
        pattern = re.sub(r'(\s|-)', r'[ -]?', term)
        search_terms.append(pattern)
    regex_patterns = [re.compile(term, re.IGNORECASE) for term in search_terms]

    # building ai_mention field
    search_columns = ['blurb','name']
    for column in search_columns:
        df['ai_matches_' + column] = df[column].apply(lambda text: [pattern.search(str(text).lower()).group() for pattern in regex_patterns if pattern.search(str(text).lower())] or None)
        df['ai_mention_' + column] = df['ai_matches_' + column].apply(lambda x: x is not None)
    df['ai_mention'] = df[['ai_mention_' + column for column in search_columns]].any(axis=1)

def data_cleaning(df, feature_names, target_name):
    # Convert all numeric columns to numeric type
    numeric_columns = df.select_dtypes(include='number').columns
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Remove NA items
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_names, how="any", inplace=True)
    df.dropna(subset=target_name, how="any", inplace=True)

def generate_plot_and_metrics():
    # future warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Use the Kickstarter dataset
    """
        ['backers_count', 'blurb', 'category', 'converted_pledged_amount', 'country', 'country_displayable_name', 
        'created_at', 'creator', 'currency', 'currency_symbol', 'currency_trailing_code', 'current_currency', 'deadline', 
        'disable_communication', 'fx_rate', 'goal', 'id', 'is_disliked', 'is_launched', 'is_liked', 'is_starrable', 
        'launched_at', 'location', 'name', 'percent_funded', 'photo', 'pledged', 'prelaunch_activated', 'profile', 
        'slug', 'source_url', 'spotlight', 'staff_pick', 'state', 'state_changed_at', 'static_usd_rate', 'urls', 
        'usd_exchange_rate', 'usd_pledged', 'usd_type', 'video']
    """
    df = pd.DataFrame()
    df_files = {}
    running_total = 0
    for fname in glob.glob(os.path.abspath('./data/**/*.csv')):
        _df=pd.read_csv(fname)
        df = df.append(_df.copy(), ignore_index=True)
        df_files[os.path.basename(fname)] = _df
        running_total+=len(_df)
        print(fname)
        print(running_total)
        #break #DEBUG
        
    data_analysis(df)

    feature_names = ['backers_count','spotlight','staff_pick','percent_funded', 'is_disliked', 'is_launched', 'is_liked', 'is_starrable', 'goal']
    target_name = 'ai_mention'

    data_cleaning(df, feature_names, target_name)

    # Start MLflow run
    with mlflow.start_run():
        X = df.loc[:, df.columns[:,None] == feature_names]
        y = df.loc[:, df.columns == target_name].values
        # Enable automatic logging to MLflow
        mlflow.set_experiment("MLflow")
        mlflow.autolog()

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

        logr_params =  {"solver": "lbfgs", "max_iter": 1, "multi_class": "auto", "random_state": 0}

        logr = LogisticRegression(**logr_params)

        # MLflow triggers logging automatically upon model fitting
        logr.fit(X_train, y_train)

        # Predict on the test set
        logr_y_pred = logr.predict(X_test)

        # Log metrics manually (optional, since autolog will capture them)
        metrics = {'MSE': 'test MSE', 'R^2': 'test r^2', 'MAE': accuracy_score(y_test, logr_y_pred)}

        logr_coefficients = logr.coef_[0]  # Obtain the coefficients

        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        plt.barh(range(len(logr_coefficients)), logr_coefficients, color='skyblue')
        # Add labels to the bars
        for i, coef in enumerate(logr_coefficients):
            plt.text(coef, i, f'{coef:.2f}', ha='left', va='center', color='black', fontsize=10)

        # Set y-axis labels and ticks
        plt.yticks(range(len(feature_names)), feature_names)

        plt.xlabel('Coefficient Value')
        plt.title('Regression Coefficients')
        plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add grid lines for better readability
        plt.show()

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

    return buf, metrics


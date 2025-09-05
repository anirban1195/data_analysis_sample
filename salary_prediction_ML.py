#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 21:50:10 2025

@author: anirban

Objective is to predict SALARY of a dev in the USA using ML
Focusing only on USA because dont want to deal with multiple currecny 
and conversion rates
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from basic_exploratory_analysis import load_and_preprocess
from statistical_tests import language_preprocess
import seaborn as sns
import shap

# Configuration
#################################################################
# Key for converting age to numbers. Take midpoint of the range
inverse_age_key = {1: 'Under 18 years old', 2: '18-24 years old', 3: '25-34 years old', 4: '35-44 years old', 
                   5: '45-54 years old', 6: '55-64 years old', 7: '65 years or older'}
#############################################################################



"""
Prepares the data for ML, feature creation, selection and further cleaning.
"""
def engineer_features(df):
    
    # Pre-process for language and create and OHE for the top 10
    # most unsed languages
    df, top_10_languages = language_preprocess(df)
    df['Age'] = df['Age'].map(inverse_age_key)
    df_ml = df.copy()
   
    
    # Select columns that are likely to be predictive of compensation
    #Add the top 10 language One Hot encoding as well 
    features_to_keep = [
        'MainBranch', 'EdLevel', 'YearsCode', 'YearsCodePro',
         'Age', 'OrgSize', 'CompTotal', 'Country', 
    ] + list(top_10_languages)
    
    df_ml = df_ml[features_to_keep]
    
    # Convert yearscode and yearscodepro to numeric type
    df_ml['YearsCodePro'] = pd.to_numeric(df_ml['YearsCodePro'], errors='coerce')
    df_ml['YearsCode'] = pd.to_numeric(df_ml['YearsCode'], errors='coerce')
    
    # --FILTER--
    # USA with reasonable salaries and pro devs
    df_ml = df_ml[(df_ml['Country'] == 'USA') & 
                  (df_ml['CompTotal']> 10000) & (df_ml['CompTotal']< 500000) &
                  (df_ml['MainBranch'] == 'Pro')].copy()
        
    
    # Drop rows with any remaining missing values in our feature set
    df_ml.dropna(inplace=True)

    #Define final features (X) and target (y)
    #We drop the categorical columns only to add them later as ONE HOT ENCODED
    categorical_cols = ['EdLevel', 'OrgSize', 'Age']
    X = df_ml.drop(columns=['CompTotal', 'Country', 'MainBranch'] + categorical_cols)
    
    
    # Create one-hot encoded features for categorical columns
    dummies = pd.get_dummies(df_ml[categorical_cols], drop_first=True)
    
    # Join the new dummy columns with the main feature DataFrame
    X = X.join(dummies)
    y = df_ml['CompTotal']

    return X, y





"""
Trains a gradient boosting model and evaluates its performance
on test data.
"""
def train_and_evaluate_model(X, y):
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Gradient Boosting Regressor model
    # with some reasonable starting hyperparameters
    gbr = GradientBoostingRegressor(
        n_estimators=70,
        learning_rate=0.15,
        max_depth=3,
        random_state=42,
        subsample=0.8
    )

    # Train the model
    print("Training Gradient Boosting model...")
    gbr.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = gbr.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(" Model Evaluation Results ")
    print(f"Root Mean Squared Error : ${rmse:,.2f}")
    print(f"R-squared: {r2:.2f}")

    return gbr, X_test # Return the model and test data for interpretation






"""
Creates feature importance to interpret the model's predictions.
"""
def interpret_model(model, X_test):
    
    # Find the models features importances
    feature_importances = model.feature_importances_
    features = X_test.columns
    
    #Make a dataframe and srot by feature importances 
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df.sort_values(by='Importance', ascending=False, inplace = True)

    # Plot importances
    plt.figure(figsize=(15, 15))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('Top 10 Most Important Features for Predicting Salary')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    
    #Load the the data and then engieer feautues for the ML algorithm 
    df, df_scheme = load_and_preprocess()
    X, y = engineer_features(df)
    
    #Train the model and evaluate
    model, X_test = train_and_evaluate_model(X, y)
    
    #See feature importance to understand what influences salary the most
    interpret_model(model, X_test)
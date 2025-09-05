#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:14:50 2025

@author: anirban
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
###########################################################
DATA_FILENAME = '/home/anirban/stack-overflow-developer-survey-2024/survey_results_public.csv'
SCHEME_FILENAME = '/home/anirban/stack-overflow-developer-survey-2024/survey_results_schema.csv'

# Key for converting age to numbers. Higher number  = Higher age bin 
age_key = {'Under 18 years old':1, '18-24 years old':2, '25-34 years old':3,
       '35-44 years old' : 4, '45-54 years old':5,'55-64 years old':6,
       '65 years or older':7}

# Key for converting LARGE country names to common names 
country_name_map = {
    'United States of America': 'USA',
    'United Kingdom of Great Britain and Northern Ireland': 'UK',
    'Peoples Republic of China': 'China',
    'Germany': 'Germany', 'India':'India','Ukraine':'Ukraine',
    'France':'France', 'Canada':'Canada','Poland':'Poland',
    'Netherlands':'Netherlands','Brazil':'Brazil'
    }

# Simple key to convert main profession to easy to understand word
profession_map ={'I am a developer by profession': 'Pro'
                 , 'I am learning to code': 'Noob',
       'I code primarily as a hobby':'Wannabe',
       'I am not primarily a developer, but I write code sometimes as part of my work/studies':'Decent',
       'I used to be a developer by profession, but no longer am': 'GotBored'}

#########################################################

'''
Module to load and preprocess data
'''
def load_and_preprocess():
    
    #Load the data and schema files
    df = pd.read_csv(DATA_FILENAME)
    df_scheme = pd.read_csv(SCHEME_FILENAME)
    
    #Convert the country and age to simpler country names and age bins
    df['Age'] = df['Age'].map(age_key)
    df['Country'] = df['Country'].map(country_name_map)
    df['MainBranch'] = df['MainBranch'].map(profession_map)
    
    #Drop rows in nans in age, country
    df.dropna(subset= ['Age', 'Country', 'MainBranch'], inplace = True)
    
    return df, df_scheme


'''
PLot histograms of ages and number of devs in the top 10 countries of the dataset 
'''
def age_country_histogram(df):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))    
    
    #Plot1 : Age counts for clean plots
    age_counts = df['Age'].value_counts().sort_index()
    axes[0].bar(age_counts.index, age_counts.values, edgecolor='black')

    # Adding Titles and Labels
    axes[0].set_title('Age Distribution of Stack Overflow Developers')
    axes[0].set_xlabel('Age Groups')
    axes[0].set_ylabel('Frequency')

    # Customizing ticks for better readability
    axes[0].set_xticks(ticks=range(1, len(age_key) + 1), labels=['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
    
    #Plot 2: Top 10 counties. No. of dev in each country
    countries = df['Country'].value_counts()
    top_ten = countries.head(10)
    top_ten.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Top 10 Most Frequent Countries of Developers')
    axes[1].set_xlabel('Countries')
    axes[1].set_ylabel('Frequency')
    axes[1].tick_params(axis='x', rotation=45)
    
    #Display in tight layout
    plt.tight_layout()
    plt.show()
    
'''
Plot a graph of what percent of dev are at what stage of their careers in each country
'''
def plot_experience_vs_country(df):
    
    #Get the df containing only the top 10 countries (i.e. top 10 most popular for devs)
    countries = df['Country'].value_counts()
    top_ten_count = countries.head(10)
    top_ten_countries = top_ten_count.index
    df_top_ten = df[df['Country'].isin(top_ten_countries)]

    #Now group by county and profession
    a = df_top_ten.groupby(['Country', 'MainBranch']).size().unstack(fill_value=0)
    percent = a.div(a.sum(axis=1), axis=0)

    # Plot a line plot
    plt.figure(figsize=(14, 8))
    
    markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'v']
    # Loop through the 'MainBranch' stages and plot each one separately as a line
    for i,stage in enumerate(percent.columns):
        #print (percent.index, percent[stage])
        plt.plot(percent.index, percent[stage], label=stage, marker=markers[i])

    # Customize the plot
    plt.title('Percentage of Developers at Different Stages of Coding by Country')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='MainBranch', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
 
    
 
'''
Plots a graph of compensation range in each age bin for a given country
'''
def plot_country_age_vs_salary(df, country_name = 'USA'):
    
    #Age ranges for different values of age. 
    #Look at the config and preprocess to get more context. 
    #We are changing age back to the original state
    numeric_to_label = {
    1: 'Under 18',
    2: '18-24',
    3: '25-34',
    4: '35-44',
    5: '45-54',
    6: '55-64',
    7: '65+'
    }
    
    # Select rows with the country and the reasonable comp range
    df_country = df[(df['Country'] == country_name) & (df['CompTotal'] > 0) & (df['CompTotal'] <5e5)]
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    
    # Plot using the numeric age column for automatic sorting
    ax = sns.boxplot(x='Age', y='CompTotal', data=df_country)
    
    #Get the labels from our rverse mapping
    age_labels = [numeric_to_label[tick] for tick in sorted(df_country['Age'].unique())]
    ax.set_xticklabels(age_labels)
    
    #Add titles and labels
    ax.set_title('Annual Compensation by Age Group' , fontsize=16)
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Annual Compensation (USD)')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
        
    
    


    
    
if __name__ == "__main__":
    
    df, df_scheme = load_and_preprocess()
    print (f'Total columns is: {len(df.columns)}')
    
    #age_country_histogram(df)
    
    #plot_experience_vs_country(df)
    
    
    #plot_country_age_vs_salary(df, country_name = 'USA')
    
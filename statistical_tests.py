#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 16:56:43 2025

@author: anirban
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from basic_exploratory_analysis import load_and_preprocess


def language_preprocess(df):
    #Find the top 10 languages used
    df['LanguageUsed'] = df['LanguageHaveWorkedWith'].dropna().str.split(';')
    language_counts = df['LanguageUsed'].explode().value_counts()
    top_10_languages = language_counts.head(10).index
    
    #We only do USA since other courrent may or may not be in USD 
    #Also use cuts in Compensation for cleaner results 
    df = df[(df['Country'] == 'USA') & (df['CompTotal'] > 0) & (df['CompTotal'] <7e5)].copy()
    df.dropna(subset= ['LanguageUsed'], inplace = True)
    
    # Create columns indicating whether a language is used by a person
    for language in top_10_languages:
        df[language] = df['LanguageUsed'].apply(lambda x: 1 if language in x else 0)

    return df, top_10_languages


"""
Calculates and plots the median salary with standard deviation error bars
for the top 10 most used programming languages among developers in the USA.
"""
def plot_language_vs_salary(df):
    
    df, top_10_languages = language_preprocess(df)
    #Dictionary to hold the median and std for each salary 
    median_salary = {}
    std_dev = {}
    for language in top_10_languages:
        median_salary[language] = df[df[language] == 1]['CompTotal'].median()
        std_dev[language] = df[df[language] == 1]['CompTotal'].std()
        
    # Assuming 'median_salary' and 'std_dev' are dictionaries with the same keys (languages)
    salary_df = pd.DataFrame({'Language': list(median_salary.keys()), 
                              'Median': list(median_salary.values()),
                              'StdDev':list(std_dev.values())})
    
    fig, ax = plt.subplots(figsize=(19, 8))
    #Plot the results
    for j in range(10):
        ax.errorbar(x=j, y=salary_df.iloc[j][1], yerr=salary_df.iloc[j][2], fmt='k.', markersize=15)
    
    #Adding labels to the plot for redability
    ax.set_xticks(ticks=range(0, 10), labels=salary_df['Language'].to_list())
    ax.set_ylabel('Median Salary (USD)')
    ax.set_title('Median Salary by Programming Language in the USA')
    ax.grid()
    
    plt.tight_layout()
    plt.show()



    
    
"""
Checks if there is a statistically significant difference between the compenstation
of users using language1 vs language2. 

"""
def anova_one_way( df, lang1, lang2):
    
    from scipy.stats import f_oneway
    df, top_10_languages = language_preprocess(df)
    
    
    print ('Top 10 used languages are')
    print (top_10_languages)
    #Calculte the p value
    f_stat, p_val = f_oneway( df[df[lang1] == 1]['CompTotal'], df[df[lang2] == 1]['CompTotal'])
    print ('++++++++++++++++++++++++++++++++++++++++++++++')
    if p_val <0.05:
        print (f'Statistically significant difference of salary found for {lang1} and {lang2}. P value is {p_val: .6f}')
    else:
        print (f'Statistically IN-significant difference of salary found for {lang1} and {lang2}. P value is {p_val: .6f}')
    print ('+++++++++++++++++++++++++++++++++++++++++++++++')

 

   
"""
Performs a Chi-Squared test of independence to check for a statistically
significant association between developer age and binned salary categories.
"""
def chi_independence(df, age_thresh = 1):  #Age thresh = 1 exludes <18 population (See exploratory_analysis.py for age mapping)
    
    from scipy import stats
    
    df, top_10_languages = language_preprocess(df)
    df = df.query('Age > @age_thresh').copy()
    
    #Divide the income in bins in multiples of US median income 
    bins = [0, 70000, 140000, 210000, 7e5]  # $70k the the median income in US
    labels =  ['Low', 'Medium', 'High', 'Very High']
    df['SalaryCategory'] = pd.cut(df['CompTotal'], bins = bins, labels=labels)
    
   
    #Create contingency table for the analysis
    cont_table = pd.crosstab(df['SalaryCategory'], df['Age'])
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(cont_table)


    # Print the results
    print(f"Chi-Squared Statistic: {chi2_stat}")
    print(f"P-Value: {p_val}")
    

    # Interpretation: If the p-value is less than 0.05, we reject the null hypothesis that there is no association.
    if p_val < 0.05:
        print("There is a statistically significant association between salary category and age category.")
    else:
        print("There is no statistically significant association between salary category and age category.")
        


"""
Performs an independent t-test to compare the mean age of developers
between two specified countries to see if the difference is statistically significant.
"""
 
def t_test(df, country1, country2):
    
    from scipy import stats
    
    #Get the ages of 2 different countries. 
    age_country1 = df[df['Country'] == country1]['Age']
    age_country2 = df[df['Country'] == country2]['Age']
    
    t_stat, p_value = stats.ttest_ind(age_country1, age_country2, equal_var=False) 
    print(f"P-Value: {p_value}")
    # Interpretation: If the p-value is less than 0.05, we reject the null hypothesis
    # Null Hypothesis: There is no difference in the age distribution of devs in country1 and country2
    if p_value < 0.05:
        print(f"There is a statistically significant difference in age distribution of devs in {country1} and {country2}")
    else:
        print(f"There is no statistically significant difference in age distribution of devs in {country1} and {country2}")
        

    
if __name__ == "__main__":
    
    df, df_scheme = load_and_preprocess()
    plot_language_vs_salary(df)
    
    
    #One way anova to compare the mean of salary distributuon of people using a given 
    #language. NOTE: People USE more than one LANGUAGE almost alwaus
    anova_one_way(df, 'C', 'C++') #Insignificant 
    anova_one_way(df, 'Python', 'JavaScript') #Significant difference
    


    #check for a statistically significant association between developer age and binned salary categories
    #using chi-square test of independence
    chi_independence(df, 1)
  
  
    
    # T test to compare the the statistical similarity of mean age of 2 countries
    t_test(df, 'Germany', 'USA')  #Difference found
    t_test(df, 'USA', 'UK')   #No difference 
    t_test(df, 'France', 'Germany')  #No difference 

############################################################################
# This is the file for correlation analysis.
# Author: Cao, Qi
# Email: q9cao@ucsd.edu
############################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

def load_data(file_path='input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv'):
    """
    load the data for correlation analysis
    """
    assert file_path.endswith('.csv')
    data = pd.read_csv(file_path)
    df = data[['TreeCoverLoss_ha', 'GrossEmissions_Co2_all_gases_Mg']]
    return df


def correlation_coefficient_computing():
    """
    compute the correlation coefficient to test the linearity between TreeLoss and CO2
    """
    df = load_data('input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv')
    pearson_corr = df.corr(method='pearson')
    spearman_corr = df.corr(method='spearman')
    print(f"person correlation coefficient:\n {pearson_corr}\n\n"
          f"spearman correlation coefficient:\n {spearman_corr}\n\n")


def scatter_plot_for_correlation_analysis():
    """
    this function provide three ways for scatter plot for correlation analysis
    1. scatter with regression line (w/o data cleaning)
    2. scatter with regression line (w/ data cleaning)
    3. scatter with regression line (w/ data cleaning and outliners eliminating)
    """
    df = load_data('input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv')
    plt.figure(figsize=(24, 6))
    # scatter with regression line
    plt.subplot(1, 3, 1)
    sns.regplot(x='TreeCoverLoss_ha', y='GrossEmissions_Co2_all_gases_Mg', data=df, scatter_kws={'alpha': 0.7})
    plt.title('TreeLoss vs CO2: scatter with regression line')
    plt.xlabel('TreeLoss/ha')
    plt.ylabel('CO2/Mg')
    plt.grid(True)
    # scatter with data cleaning
    plt.subplot(1, 3, 2)
    small = df[df['TreeCoverLoss_ha'] < 0.1 * 1e6]
    sns.regplot(x='TreeCoverLoss_ha', y='GrossEmissions_Co2_all_gases_Mg', data=small, scatter_kws={'alpha': 0.7})
    plt.title('TreeLoss vs CO2: scatter with data cleaning')
    plt.xlabel('TreeLoss/ha')
    plt.ylabel('CO2/Mg')
    plt.grid(True)
    # scatter with data cleaning and outliners eliminating
    plt.subplot(1, 3, 3)
    small = df[df['TreeCoverLoss_ha'] < 0.1 * 1e5]
    sns.regplot(x='TreeCoverLoss_ha', y='GrossEmissions_Co2_all_gases_Mg', data=small, scatter_kws={'alpha': 0.7})
    plt.title('TreeLoss vs CO2: scatter with data cleaning and outliners eliminating')
    plt.xlabel('TreeLoss/ha')
    plt.ylabel('CO2/Mg')
    plt.grid(True)

    plt.show()


def OLS_function_for_correlation_analysis():
    """
    the ordinary least squares regression line (OLS) function for correlation analysis:
    TreeLoss is X, and CO2 is Y
    """
    df = load_data()
    small = df[df['TreeCoverLoss_ha'] < 0.1 * 1e6]
    y = small['GrossEmissions_Co2_all_gases_Mg']
    X = small['TreeCoverLoss_ha']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    params = model.params
    equation = f"GrossEmissions_Co2_all_gases_Mg = {params['const']:.2f}"
    for param, value in params.items():
        if param != 'const':
            equation += f" + ({value:.2f}) * {param}"

    print("\nOrdinary least squares regression line (OLS) function:")
    print(equation)

def regression_error_visualization():
    """
    visualize the regression error between TreeLoss and CO2
    """
    df = load_data()
    small = df[df['TreeCoverLoss_ha'] < 0.1 * 1e6]
    y = small['GrossEmissions_Co2_all_gases_Mg']
    X = small['TreeCoverLoss_ha']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # predict
    y_pred = model.predict(X)

    # residual
    residuals = y - y_pred

    # plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('predict value')
    plt.ylabel('error value')
    plt.title('predict value vs error value')
    plt.show()


def upperbound_regression_error_visualization():
    """
    visualize the upperbound of regression error between TreeLoss and CO2
    """
    df = load_data()
    small = df[df['TreeCoverLoss_ha'] < 0.1 * 1e6]
    y = small['GrossEmissions_Co2_all_gases_Mg']
    X = small['TreeCoverLoss_ha']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # predict
    y_pred = model.predict(X)

    # residual
    residuals = y - y_pred

    # plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('predict value')
    plt.ylabel('error value')
    plt.title('predict value vs error value')
    # y = 0.4x
    x_line = np.linspace(y_pred.min(), y_pred.max(), 100)
    y_line = 0.45 * x_line
    plt.plot(x_line, y_line, label='y = 0.4x', color='green', linestyle='--', linewidth=2)
    # y = -0.4x
    x_line = np.linspace(y_pred.min(), y_pred.max(), 100)
    y_line = -0.45 * x_line
    plt.plot(x_line, y_line, label='y = 0.4x', color='green', linestyle='--', linewidth=2)

    plt.show()









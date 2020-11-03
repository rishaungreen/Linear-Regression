import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pylab
import math
import pickle
import statsmodels.api as sm

from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew


# Open the dataframe
"""Navigate to the file that holds the data"""
original_file = 'D:/DATA/US DATA/Master DATA 2.xlsx'

"""Input the data from the file into a dataframe"""
df = pd.read_excel(original_file)


# Check data sample
"""View the dataframe, to check it has imported correctly"""
print()
print('US Socioeconomics against Property crime by state in 2014')
print('-'*100)
print('-'*100)
print(df.head(20))
print()
print()


# Remove extraneous columns
"""Remove the year column as it is not used in the model"""
df = df.drop(['Year'], axis = 1)


# Change Index
"""Change the data to be indexed by state"""
df.index = df['State']
df = df.drop(['State'], axis = 1)


# Check for missing values
"""Check if there are any missing or nil values that have to be excluded from the model"""
print()
print('Assess the presence of any missing or nil values')
print('-'*100)
print('-'*100)
if df.isna().any().any() == True:
    print('There are some missing values')
else:
    print('There are no missing values')
print()
print()


# Check types
"""Ensure that, following the removal of unneccessary columns, the columns are of the correct type (float in this case)"""
print()
print('The data type for each column in the dataframe')
print('-'*100)
print('-'*100)
print(df.dtypes)
print()
print()


# Rename Columns for ease of use going forward
new_column_names = {'Population': 'POP_TOT',
                    'GDP (per capita)': 'GDP',
                    'Property crime (per 10000 people)': 'CRIME'}
df = df.rename(columns = new_column_names)


# Check for Multicolinearity
"""Check that each of the columns are not substantially impacting/predetermining the values in another"""
#corr = df.corr()
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='PuOr_r')
#plt.show()
df_before = df
X1 = sm.tools.add_constant(df_before)
series_before = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index = X1.columns)
print()
print("Original Dataframe")
print('Checking which columns return high VIF values (over 4)')
print('-'*100)
print('-'*100)
print(series_before)
print()
print()
#df = df.drop(['POP_M', 'POP_F', 'POP_0+', 'POP_46+'], axis = 1)
#X2 = sm.tools.add_constant(df)
#series_after = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index = X2.columns)
#print()
#print("Altered Dataframe")
#print('Checking that there are no further high VIF values')
#print('-'*100)
#print('-'*100)
#print(series_after)
#print()
#print()


# Check for Outliers
"""Identify values that are more than 3 standard deviations from the mean"""
desc_data =  df.describe()
desc_data.loc['+3_std'] = desc_data.loc['mean'] + (desc_data.loc['std']*3)
desc_data.loc['-3_std'] = desc_data.loc['mean'] - (desc_data.loc['std']*3)
print()
print(desc_data)
df_remove_data = df[(np.abs(stats.zscore(df)) < 3).all(axis = 1)]
df_outlier_index = df.index.difference(df_remove_data.index)
print()
print('The data contains ' + f'{len(df_outlier_index)}' + ' outliers:')
print(df_outlier_index)
print()
print()


# Create Regression model
""" Split the data """
X = df.drop('CRIME', axis = 1)
Y = df[['CRIME']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

""" Create an instance of the model """
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
intercept = regression_model.intercept_[0]
coef = regression_model.coef_[0]
print()
print('Checking the intercept and coefficient values')
print('-'*100)
print('-'*100)
print('The intercept value is ' + f'{intercept}')
print()
for coef in zip(X.columns, regression_model.coef_[0]):
    print("The Coefficient for " + f"{coef[0]}" + " is " + f"{coef[1]:.10}")
print()
print()

""" Refitting model for evaluation """
y_predict = regression_model.predict(X_test)
X2 = sm.add_constant(X)
model = sm.OLS(Y, X2)
est = model.fit()
est.conf_int()
est.pvalues


#Checking for Heteroscedasticity
"""Consistent variance along the regression line"""
_, pval, _, f_pval = diag.het_white(est.resid, est.model.exog)
print(pval, f_pval)
print()
if pval > 0.05:
    print("For the Heteroscedasticity Test")
    print('-'*100)
    print('-'*100)
    print("The p-value was greater than 0.05 at " +  f"{pval:.4}")
    print("We fail to reject the null hypthoesis, so there is no heterosecdasticity. \n")    
else:
    print("For the Heteroscedasticity Test")
    print('-'*100)
    print('-'*100)
    print("The p-value was smaller than 0.05 at " +  f"{pval:.4}")
    print("We reject the null hypthoesis, so there is heterosecdasticity. \n")
print()
print()


# Check for Autocorrelation
"""Check to see if terms can effectively predict the terms that succeed them"""
print()
lag = min(10, (len(X)//5))
print("Checking the probability of autocorrelation is lower than 5%")
print("The number of lags will be " + f"{lag}")
print('-'*100)
print('-'*100)
auto_corr = diag.acorr_ljungbox(est.resid, lags = lag)
ibvalue, p_val = auto_corr
if min(p_val) > 0.05:
    print("The smallest p-value was greater than 0.05 at " +  f"{min(p_val):.4}")
    print("We fail to reject the null hypthoesis, so there is no autocorrelation.")
else:
    print("The smallest p-value was smaller than 0.05 at " +  f"{min(p_val):.4}")
    print("We reject the null hypthoesis, so there is autocorrelation.")
print()
print()
"""There is some autocorrelation probably, because of the use of successive years (Solution: remove all but one year?)"""


# Check for Normal Distribution (mean of residuals = 0)
sm.qqplot(est.resid, line = 's')
pylab.show()
mean_residuals = sum(est.resid) / len(est.resid)
print()
print("Checking the data is normally distributed (mean of residuals ~ 0)")
print('-'*100)
print('-'*100)
print(mean_residuals)
print()
print()


# Measure the error
model_mse = mean_squared_error(y_test, y_predict)
model_mae = mean_absolute_error(y_test, y_predict)
model_rmse = math.sqrt(model_mse)
print()
print("Checking error values")
print('-'*100)
print('-'*100)
print("Mean Absolute Error: " + f"{model_mae}")
print("Mean Squared Error: " + f"{model_mse}")
print("Root Mean Squared Error: " + f"{model_rmse}")
print()

""" R-Squared """
model_r2 = r2_score(y_test, y_predict)
adj_r2 = (1 - (1 - model_r2) * ((len(X_train) - 1) / (len(X_train) - X.shape[1] - 1)))
print()
print("Checking R-squared values")
print('-'*100)
print('-'*100)
print("R-Squared Value: " + f"{model_r2}")
print("Adjusted R-Squared Value: " + f"{adj_r2}")

"""Pickle the Model"""
with open('my_multilinear_regression.sav', 'wb') as f:
    pickle.dump(regression_model, f)
with open('my_multilinear_regression.sav', 'rb') as pickle_file:
    regression_model_2 = pickle.load(pickle_file)

os.system("pause")

# Operation that will predict the dependent variable based on the model and input values for independent variables
print()
print()
while True:
    os.system('cls')            
    val_1 = float(input("Population: "))
    val_2 = float(input("GDP (per capita): "))
    data01 = {'POP_TOT': [val_1], 'GDP': [val_2]}
    data02 = pd.DataFrame (data01, columns = ['POP_TOT', 'GDP'])
    result = regression_model_2.predict(data02)[0][0]
    print("If the GDP (per capita) is " + f"{val_2}" + " and the population is " + f"{val_1}")
    print("The expected instances of property crime (per 10000 people) is " + f"{result:.2f}")
    os.system("pause")

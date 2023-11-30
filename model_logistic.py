# region: Import dependencies

import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# endregion

# region: Load cleansed data

filename = 'bicycle_thefts_data_regr.csv'
filepath = os.getcwd() + '/data/' + filename
bicycle_thefts_data_regr = pd.read_csv(filepath)

# endregion

# region: TODO: Logistic Regression Modeling - Predicting if the bike is stolen or not

# TODO: Feature Selection (Backward selection to keep the variables with low p-values and eliminate the ones with high p-values to increase the R squared value)

# TODO: Split data into training and testing sets (75:25)

# TODO: Train the model (Managing imbalanced classes if needed)

# TODO: Model Scoring and Evaluation (Confusion Matrix)


# endregion


# region: TODO: Decision Tree Modeling - Predicting if the bike is stolen or not

# TODO: Feature Selection (Backward selection to keep the variables with low p-values and eliminate the ones with high p-values to increase the R squared value)

# TODO: Split data into training and testing sets (75:25)

# TODO: Train the model (Managing imbalanced classes if needed)

# TODO: Model Scoring and Evaluation

# endregion

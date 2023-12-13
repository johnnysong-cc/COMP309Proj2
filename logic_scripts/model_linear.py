# region: Import dependencies

import os, joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns
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
filepath = os.path.dirname(os.getcwd()) + '/data/' + filename
bicycle_thefts_data_regr = pd.read_csv(filepath)

# endregion

# region: Linear Regression Modeling - Predicting report delay in hours based on the other variables

# Feature Selection
bicycle_thefts_data_regr['Report_Delay'] = pd.to_datetime((bicycle_thefts_data_regr['REPORT_TIMESTAMP'] - bicycle_thefts_data_regr['OCC_TIMESTAMP']), unit='s').dt.hour
# data_encoded = bicycle_thefts_data_regr\
#                 .select_dtypes(include=[np.number])\
#                 .drop(['OCC_TIMESTAMP', 'REPORT_TIMESTAMP', 'BIKE_SPEED', 'BIKE_COST'], axis=1)
data_encoded_ori = pd.get_dummies(bicycle_thefts_data_regr.drop(['EVENT_UNIQUE_ID','LOCATION_TYPE','OCC_TIMESTAMP','REPORT_TIMESTAMP','BIKE_COST','BIKE_SPEED'], axis=1),drop_first=True)
data_encoded = data_encoded_ori.astype(float)
print(data_encoded.columns)
X = data_encoded.drop(['BIKE_COST_NORMALIZED'], axis=1)
y = data_encoded['BIKE_COST_NORMALIZED']
X = sm.add_constant(X)  # add a constant to the model for the intercept
linear_model = sm.OLS(y, X).fit()
print(linear_model.summary())

p_max=1
while p_max>0.05:
    p_values = linear_model.pvalues
    p_max = p_values.max()
    feature_max_p = p_values.idxmax()
    if p_max>0.05:
        X.drop(feature_max_p, axis=1, inplace=True)
        linear_model = sm.OLS(y, X).fit()

print(X.columns)

# Split data into training and testing sets (75:25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Model Scoring and Evaluation (R squared value)
print(f'\nCoefficients: {linear_model.coef_}')
print(f'\nIntercept: {linear_model.intercept_}')
print(list(zip(X, linear_model.coef_)))
print(f'\nR-squared: {linear_model.score(X_test, y_test)}')
print(f'\nscore: {linear_model.score(X_test, y_test)}')
y_pred = linear_model.predict(X_test)
y_pred_unnormalized = y_pred * bicycle_thefts_data_regr['BIKE_COST'].std() + bicycle_thefts_data_regr['BIKE_COST'].mean()
print(f"\npredicted: {y_pred_unnormalized}", f"\nmean is: {bicycle_thefts_data_regr['BIKE_COST'].mean()}", f"\nstd is: {bicycle_thefts_data_regr['BIKE_COST'].std()}")

# endregion

# region: export model
joblib.dump(linear_model, '../models/linear_model.pkl')
joblib.dump(X.columns, '../models/linear_model_columns.pkl')
print("Models columns dumped!")
# endregion

# region: Import dependencies

import os
import sys
import random
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import linear_model
from sklearn import metrics

# endregion

# region: Load cleansed data

filename = 'bicycle_thefts_data_regr.csv'
filepath = os.getcwd() + '/data/' + filename
bicycle_thefts_data_regr = pd.read_csv(filepath)

# endregion

# region: Logistic Regression Modeling - Predicting if the bike is stolen or not
bicycle_thefts_data_regr['STATUS'] = bicycle_thefts_data_regr['STATUS'].apply(lambda x: 1 if x == 'STOLEN' else 0)
bicycle_thefts_data_regr['OCC_MONTH'] = pd.to_datetime(bicycle_thefts_data_regr['OCC_TIMESTAMP'], unit='s').dt.month
data_encoded = bicycle_thefts_data_regr.drop(['EVENT_UNIQUE_ID','LOCATION_TYPE','OCC_TIMESTAMP','REPORT_TIMESTAMP','BIKE_COST','BIKE_SPEED'], axis=1)

# region: Check dataset
# print(data_encoded.columns)
# print(data_encoded.groupby('STATUS').count())
# print(data_encoded.groupby('STATUS').mean(numeric_only=True))
# table1 = pd.crosstab(data_encoded['OCC_MONTH'], data_encoded['STATUS'])
# table2 = pd.crosstab(data_encoded['PREMISES_TYPE'], data_encoded['STATUS'])
# table3 = pd.crosstab(data_encoded['BIKE_MAKE'], data_encoded['STATUS'])
# table4 = pd.crosstab(data_encoded['BIKE_COLOUR'], data_encoded['STATUS'])
# table1.div(table1.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(4,4))
# table2.div(table2.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(4,4))
# table3.div(table3.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(4,4))
# table4.div(table4.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(4,4))
# plt.show()
# endregion

# region: dummy variables
for var in data_encoded.columns:
    if data_encoded[var].dtype == 'object':
        cat_list = pd.get_dummies(data_encoded[var], prefix=var)
        data_encoded = data_encoded.join(cat_list)
        data_encoded.drop(var, axis=1, inplace=True) # drop the original categorical variable

print(f'\n{data_encoded.columns}\nNumber of columns: {len(data_encoded.columns)}')
# endregion




# Feature Selection (Backward selection to keep the variables with low p-values and eliminate the ones with high p-values to increase the R squared value)

X = data_encoded.drop(['STATUS'], axis=1)
Y = data_encoded['STATUS'].ravel()
model = LogisticRegression(max_iter=5000)
rfe = RFE(model, n_features_to_select=15)
rfe = rfe.fit(X, Y)
print(rfe.support_)
print(rfe.ranking_)
selected_features = X.columns[rfe.support_]
print(selected_features)


# Split data into training and testing sets (75:25)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# Train the model (Managing imbalanced classes if needed)
clf1=linear_model.LogisticRegression(solver='lbfgs')
clf1.fit(X_train, y_train)


# Model Scoring and Evaluation (Confusion Matrix)
probs=clf1.predict_proba(X_test)
print(f'\nProbabilities:\n{probs}')
predicted = clf1.predict(X_test)
print(f'\nPredicted:\n{predicted}')
print(f'\nR-squared: {clf1.score(X_test, y_test)}')

scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs',max_iter=5000), X, Y, scoring='accuracy', cv=10)
print(f'\nScores:\n{scores}')
print(f'\nMean Score: {scores.mean()}')

prob = probs[:, 1]
prob_df = pd.DataFrame(prob)
prob_df['predict'] = np.where(prob_df[0] >= 0.05, 1, 0)
Y_P = np.array(prob_df['predict'])
confusion_matrix = metrics.confusion_matrix(y_test, Y_P)
print(f'\nConfusion Matrix:\n{confusion_matrix}')

# endregion

# region: dump the model
joblib.dump(clf1, './models/model_logistic.pkl')
print("Models columns dumped!")
# endregion

# region: TODO: Decision Tree Modeling - Predicting if the bike is stolen or not

# TODO: Feature Selection (Backward selection to keep the variables with low p-values and eliminate the ones with high p-values to increase the R squared value)

# TODO: Split data into training and testing sets (75:25)

# TODO: Train the model (Managing imbalanced classes if needed)

# TODO: Model Scoring and Evaluation

# endregion
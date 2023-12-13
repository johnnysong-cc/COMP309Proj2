# region: Import dependencies

import os,sys,random,joblib
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.feature_selection import RFE
from sklearn import linear_model
from sklearn import metrics


# endregion

# region: Load cleansed data
filename = 'bicycle_thefts_data_regr.csv'
filepath = os.path.dirname(os.getcwd()) + '/data/' + filename
bicycle_thefts_data_regr = pd.read_csv(filepath)
# endregion

# region: categorical variable coding
bicycle_thefts_data_regr['STATUS'] = bicycle_thefts_data_regr['STATUS'].apply(lambda x: 1 if x == 'STOLEN' else 0)
bicycle_thefts_data_regr['OCC_MONTH'] = pd.to_datetime(bicycle_thefts_data_regr['OCC_TIMESTAMP'], unit='s').dt.month
data_encoded = bicycle_thefts_data_regr.drop(['EVENT_UNIQUE_ID', 'LOCATION_TYPE', 'OCC_TIMESTAMP', 'REPORT_TIMESTAMP', 'BIKE_COST', 'BIKE_SPEED'], axis=1)

for var in data_encoded.columns:
  if data_encoded[var].dtype == 'object':
    cat_list = pd.get_dummies(data_encoded[var], prefix=var)
    data_encoded = data_encoded.join(cat_list)
    data_encoded.drop(var, axis=1, inplace=True) # drop the original categorical variable

print(f'\n{data_encoded.columns}\nNumber of columns: {len(data_encoded.columns)}')
# endregion

# region: feature selection
X = data_encoded.drop(['STATUS'], axis=1)
Y = data_encoded['STATUS'].ravel()
model = LogisticRegression(max_iter=5000)
rfe = RFE(model, n_features_to_select=15)
rfe = rfe.fit(X, Y)
print(rfe.support_)
print(rfe.ranking_)
selected_features = X.columns[rfe.support_]
print(selected_features)
# endregion

# region: split data into training and testing sets (75:25)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
# endregion

# region: train the model
clf1 = linear_model.LogisticRegression(solver='lbfgs')
clf1.fit(X_train, y_train)
# endregion

# region: score and evaluate the model
probs = clf1.predict_proba(X_test)
print(f'\nProbabilities:\n{probs}')
predicted = clf1.predict(X_test)
print(f'\nPredicted:\n{predicted}')
print(f'\nR-squared: {clf1.score(X_test, y_test)}')

scores = cross_val_score(linear_model.LogisticRegression(
    solver='lbfgs', max_iter=5000), X, Y, scoring='accuracy', cv=10)
print(f'\nScores:\n{scores}')
print(f'\nMean Score: {scores.mean()}')

prob = probs[:, 1]
prob_df = pd.DataFrame(prob)
prob_df['predict'] = np.where(prob_df[0] >= 0.05, 1, 0)
Y_P = np.array(prob_df['predict'])
confusion_matrix = metrics.confusion_matrix(y_test, Y_P)
print(f'\nConfusion Matrix:\n{confusion_matrix}')

y_score = clf1.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf1.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

# endregion

if not os.path.exists('../models'):
    os.mkdir('../models')

# region: dump the model
joblib.dump(clf1, '../models/model_logistic.pkl')
joblib.dump(selected_features, '../models/model_logistic_features.pkl')
print("Models columns dumped!")
# endregion

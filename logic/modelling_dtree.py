# region: Import dependencies

import os
import sys
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# endregion

# region: Decision Tree Modeling - Predicting if the bike is stolen or not

# Load the data
filename = 'bicycle_thefts_data_regr.csv'
filepath = os.path.dirname(os.getcwd()) + '/data/' + filename
bicycle_thefts_data = pd.read_csv(filepath)

# Convert 'STATUS' to binary (1 for 'STOLEN', 0 for other statuses)
bicycle_thefts_data['STATUS'] = bicycle_thefts_data['STATUS'].apply(lambda x: 1 if x == 'STOLEN' else 0)

# Selecting features for the model - Dropping non-numeric and target variable
features = bicycle_thefts_data.select_dtypes(include=[np.number]).drop(columns=['STATUS'])

# Target variable
target = bicycle_thefts_data['STATUS']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Creating the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Making predictions
y_pred = decision_tree.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Save the model
joblib.dump(decision_tree, '../models/decision_tree_model.pkl')

# endregion

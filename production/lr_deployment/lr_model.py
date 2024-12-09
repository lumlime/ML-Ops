import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
parser.add_argument("--testdata", type=str, required=True, help='Dataset for testing')
args = parser.parse_args()
mlflow.autolog()



# Load training and test data
train_df = pd.read_csv('args.trainingdata')
test_df = pd.read_csv('args.testdata')

# 'Activity' is the column with class labels
Y_train = train_df['Activity'].values
X_train = train_df.drop(columns=['Activity']).values

Y_test = test_df['Activity'].values
X_test = test_df.drop(columns=['Activity']).values

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Important: use the same scaler without refitting

# Initialize Logistic Regression model
logistic_model = LogisticRegression(C=1/0.1, solver="liblinear", random_state=42)

# Fit the model on training data
logistic_model.fit(X_train, Y_train)

# Predictions on test data
logistic_predictions = logistic_model.predict(X_test)

# Performance Evaluation
print("Logistic Regression Accuracy:", accuracy_score(Y_test, logistic_predictions))

# Calculate ROC AUC score
# Ensure predict_proba is used for ROC AUC calculation
logistic_probs = logistic_model.predict_proba(X_test)
logistic_roc_auc = roc_auc_score(Y_test, logistic_probs, multi_class='ovr')
print("Logistic Regression ROC AUC:", logistic_roc_auc)

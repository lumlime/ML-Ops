import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str, required=True, help='Dataset for training')
parser.add_argument("--test_data", type=str, required=True, help='Dataset for testing')
args = parser.parse_args()

def load_data():
    """Load training and testing data from the given paths."""
    train_df = pd.read_csv(args.training_data)
    test_df = pd.read_csv(args.test_data)
    
    Y_train = train_df['Activity'].values
    X_train = train_df.drop(columns=['Activity']).values
    
    Y_test = test_df['Activity'].values
    X_test = test_df.drop(columns=['Activity']).values
    
    return X_train, Y_train, X_test, Y_test

def preprocess_data(X_train, X_test):
    """Standardize the data using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def feature_selection(X_train, Y_train, X_test, num_features=10):
    """Select top features based on ANOVA F-statistic."""
    selector = SelectKBest(score_func=f_classif, k=num_features)
    X_train_selected = selector.fit_transform(X_train, Y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected

def train_model(X_train, Y_train, hyperparameter_tuning=False):
    """Train the Logistic Regression model, with optional hyperparameter tuning."""
    if hyperparameter_tuning:
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(LogisticRegression(solver="liblinear", random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = LogisticRegression(C=1/0.1, solver="liblinear", random_state=42)
        best_model.fit(X_train, Y_train)
    return best_model

def evaluate_model(model, X_test, Y_test):
    """Evaluate the model using accuracy and ROC AUC score."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    probabilities = model.predict_proba(X_test)
    roc_auc = roc_auc_score(Y_test, probabilities, multi_class='ovr')
    
    print("Logistic Regression Accuracy:", accuracy)
    print("Logistic Regression ROC AUC:", roc_auc)
    return accuracy, roc_auc

def main():
    
    mlflow.autolog()

    # Step 1: Load Data
    X_train, Y_train, X_test, Y_test = load_data()

    # Step 2: Preprocess Data
    X_train, X_test = preprocess_data(X_train, X_test)

    # Step 3: Feature Selection (optional improvement)
    X_train, X_test = feature_selection(X_train, Y_train, X_test, num_features=10)

    # Step 4: Train Model
    model = train_model(X_train, Y_train, hyperparameter_tuning=True)

    # Step 5: Evaluate Model
    evaluate_model(model, X_test, Y_test)

if __name__ == "__main__":
    # Specify file paths directly
    main()

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_disease_model():
  print('--- Training Disease Classification Model ---')
  # Load Disease/Breast Cancer dataset as a standard classification task
  data = load_breast_cancer()
  X, y = data.data, data.target

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Start MLflow Run for Logistic Regression
  with mlflow.start_run(run_name='Disease_LogisticRegression'):
    # Hyperparameters
    C_param = 1.0
    max_iter = 200

    mlflow.log_param('model_type', 'LogisticRegression')
    mlflow.log_param('C', C_param)
    mlflow.log_param('max_iter', max_iter)

    # Model Training
    model = LogisticRegression(C=C_param, max_iter=max_iter)
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f'Logistic Regression Accuracy: {acc:.4f}')
    mlflow.log_metric('accuracy', acc)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, 'logistic_regression_model')
    print('Disease model logged to MLflow successfully.\n')


def train_boston_model():
  print('--- Training Boston Housing Regression Model ---')
  # Fetch Boston Housing dataset safely using openml
  boston = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
  X = boston.data.dropna()
  y = boston.target.loc[X.index]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Start MLflow Run for Random Forest
  with mlflow.start_run(run_name='Boston_RandomForest'):
    # Hyperparameters
    n_estimators = 100
    max_depth = 10

    mlflow.log_param('model_type', 'RandomForestRegressor')
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)

    # Model Training
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f'Random Forest MSE: {mse:.4f}, R2 Score: {r2:.4f}')
    mlflow.log_metric('mse', mse)
    mlflow.log_metric('r2_score', r2)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, 'random_forest_model')
    print('Boston Housing model logged to MLflow successfully.\n')


if __name__ == '__main__':
  # Set tracking URI (can be local folder or remote server)
  mlflow.set_tracking_uri('sqlite:///mlflow.db')
  mlflow.set_experiment('MLOps_Project_Experiment')

  train_disease_model()
  train_boston_model()
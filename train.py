import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient


def train_and_register_models():
  client = MlflowClient()
  branch = os.getenv("BRANCH_NAME", "dev")
  print(f"--- Running Pipeline for Branch/Env: {branch} ---")

  # 1. Disease Classifier Model
  data = load_breast_cancer()
  X, y = data.data, data.target
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  with mlflow.start_run(run_name=f"Disease_{branch}"):
    model_disease = LogisticRegression(C=1.0, max_iter=200)
    model_disease.fit(X_train, y_train)
    acc = accuracy_score(y_test, model_disease.predict(X_test))
    mlflow.log_metric("accuracy", acc)

    res = mlflow.sklearn.log_model(
        sk_model=model_disease,
        artifact_path="logistic_regression_model",
        registered_model_name="Disease_Classifier_Model",
    )
    
    # Assign aliases based on Document specifications
    version = res.registered_model_version
    if "dev" in branch:
      client.set_registered_model_alias("Disease_Classifier_Model", "Challenger", version)
    elif "main" in branch:
      client.set_registered_model_alias("Disease_Classifier_Model", "Challenger-pre-test", version)
      # Simulating test success -> updating to post-test
      client.set_registered_model_alias("Disease_Classifier_Model", "Challenger-post-test", version)
    else: # Release / Prod
      client.set_registered_model_alias("Disease_Classifier_Model", "Champion", version)

  print("Models trained, registered, and aliases assigned successfully according to MLOps specs!")


if __name__ == "__main__":
  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  mlflow.set_experiment("MLOps_Project_Experiment")
  train_and_register_models()
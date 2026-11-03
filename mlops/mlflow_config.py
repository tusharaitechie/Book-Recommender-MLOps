
import mlflow
import os


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

EXPERIMENT_NAME = "Book-Recommender-Experiments"


def configure_mlflow():

    # Configure tracking database
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create or select experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("===================================")
    print("MLflow Configuration Successful")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("===================================")

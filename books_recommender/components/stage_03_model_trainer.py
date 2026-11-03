
import os
import sys
import pickle
import time

import mlflow

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException


class ModelTrainer:

    def __init__(self, app_config=AppConfiguration()):

        try:
            self.model_trainer_config = (
                app_config.get_model_trainer_config()
            )

        except Exception as e:
            raise AppException(e, sys) from e

    def train(self):

        try:
            start_time = time.time()

            # Load transformed data
            with open(
                self.model_trainer_config.transformed_data_file_dir,
                "rb"
            ) as file:

                book_pivot = pickle.load(file)

            # Convert data into sparse matrix
            book_sparse = csr_matrix(book_pivot)

            # Model parameters
            model_params = {
                "algorithm": "brute",
                "n_neighbors": 5,
                "metric": "cosine"
            }

            # Log parameters
            mlflow.log_params(model_params)

            # Train model
            model = NearestNeighbors(
                algorithm=model_params["algorithm"],
                n_neighbors=model_params["n_neighbors"],
                metric=model_params["metric"]
            )

            model.fit(book_sparse)

            # Calculate dataset metrics
            n_books, n_users = book_pivot.shape

            total_values = n_books * n_users

            non_zero_values = int(
                book_sparse.nnz
            )

            sparsity = (
                1 - non_zero_values / total_values
                if total_values else 0
            )

            training_time = time.time() - start_time

            # Log metrics
            mlflow.log_metrics({
                "n_books": float(n_books),
                "n_users": float(n_users),
                "non_zero_values": float(non_zero_values),
                "sparsity": float(sparsity),
                "training_time_seconds": float(training_time)
            })

            # Save trained model locally
            os.makedirs(
                self.model_trainer_config.trained_model_dir,
                exist_ok=True
            )

            file_name = os.path.join(
                self.model_trainer_config.trained_model_dir,
                self.model_trainer_config.trained_model_name
            )

            with open(file_name, "wb") as file:
                pickle.dump(model, file)

            logging.info(
                f"Saving final model to {file_name}"
            )

            # Log model artifact to MLflow
            mlflow.log_artifact(
                file_name,
                artifact_path="trained_model"
            )

            # Log dataset information
            mlflow.set_tag(
                "model_type",
                "NearestNeighbors"
            )

            mlflow.set_tag(
                "training_status",
                "completed"
            )

            logging.info(
                "Model training and MLflow logging completed."
            )

            return file_name

        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_model_trainer(self):

        try:

            logging.info(
                f"{'='*20}Model Trainer log started.{'='*20}"
            )

            model_path = self.train()

            logging.info(
                f"{'='*20}Model Trainer log completed.{'='*20}"
            )

            return model_path

        except Exception as e:
            raise AppException(e, sys) from e

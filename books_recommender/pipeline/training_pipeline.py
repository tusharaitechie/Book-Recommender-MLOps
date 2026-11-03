
import mlflow
import time

from books_recommender.components.stage_00_data_ingestion import DataIngestion
from books_recommender.components.stage_01_data_validation import DataValidation
from books_recommender.components.stage_02_data_transformation import DataTransformation
from books_recommender.components.stage_03_model_trainer import ModelTrainer

from mlops.mlflow_config import configure_mlflow


class TrainingPipeline:

    def __init__(self):

        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_training_pipeline(self):

        configure_mlflow()

        with mlflow.start_run(
            run_name="Book-Recommender-Training"
        ):

            try:

                mlflow.set_tags({
                    "project": "Book-Recommender-MLOps",
                    "author": "Tushar Nile",
                    "pipeline": "End-to-End Training",
                    "framework": "Scikit-Learn"
                })

                pipeline_start = time.time()

                # Stage 00 - Data Ingestion
                stage_start = time.time()

                self.data_ingestion.initiate_data_ingestion()

                mlflow.log_metric(
                    "data_ingestion_time",
                    time.time() - stage_start
                )

                # Stage 01 - Data Validation
                stage_start = time.time()

                self.data_validation.initiate_data_validation()

                mlflow.log_metric(
                    "data_validation_time",
                    time.time() - stage_start
                )

                # Stage 02 - Data Transformation
                stage_start = time.time()

                self.data_transformation.initiate_data_transformation()

                mlflow.log_metric(
                    "data_transformation_time",
                    time.time() - stage_start
                )

                # Stage 03 - Model Training
                stage_start = time.time()

                self.model_trainer.initiate_model_trainer()

                mlflow.log_metric(
                    "model_training_time",
                    time.time() - stage_start
                )

                # Total pipeline execution time
                total_time = time.time() - pipeline_start

                mlflow.log_metric(
                    "total_pipeline_time",
                    total_time
                )

                mlflow.set_tag(
                    "pipeline_status",
                    "completed"
                )

                print("===================================")
                print("Training Pipeline Completed!")
                print("MLflow Tracking Successful!")
                print("===================================")

            except Exception as e:

                mlflow.set_tag(
                    "pipeline_status",
                    "failed"
                )

                raise e

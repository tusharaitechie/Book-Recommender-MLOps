
import argparse

from books_recommender.components.stage_00_data_ingestion import DataIngestion
from books_recommender.components.stage_01_data_validation import DataValidation
from books_recommender.components.stage_02_data_transformation import DataTransformation
from books_recommender.components.stage_03_model_trainer import ModelTrainer


def run_stage(stage):

    if stage == "data_ingestion":
        DataIngestion().initiate_data_ingestion()

    elif stage == "data_validation":
        DataValidation().initiate_data_validation()

    elif stage == "data_transformation":
        DataTransformation().initiate_data_transformation()

    elif stage == "model_training":
        ModelTrainer().initiate_model_trainer()

    else:
        raise ValueError(f"Unknown stage: {stage}")

    print(f"{stage} completed successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "data_ingestion",
            "data_validation",
            "data_transformation",
            "model_training"
        ]
    )

    args = parser.parse_args()

    run_stage(args.stage)


import mlflow


def start_experiment():

    mlflow.start_run(
        run_name="Book-Recommender-Training"
    )


def log_pipeline_info():

    mlflow.set_tag(
        "project",
        "Book-Recommender-MLOps"
    )

    mlflow.set_tag(
        "author",
        "Tushar Nile"
    )

    mlflow.set_tag(
        "pipeline",
        "End-to-End Book Recommendation"
    )

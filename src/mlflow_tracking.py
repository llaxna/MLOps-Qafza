from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "artifacts/models/final_model.joblib"

mlflow.set_tracking_uri(
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

mlflow.set_experiment("delivery_late_prediction")


def track_model():
    model = joblib.load(MODEL_PATH)

    with mlflow.start_run(run_name="task2_final_model"):

        # Parameters from Task 2
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", 10)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)

        # Test metrics from Task 2
        mlflow.log_metric("test_f1", 0.1709)
        mlflow.log_metric("test_recall", 0.8704)
        mlflow.log_metric("test_precision", 0.0948)
        mlflow.log_metric("test_roc_auc", 0.6733)
        mlflow.log_metric("test_pr_auc", 0.1156)

        # Save the existing model to MLflow
        mlflow.sklearn.log_model(
            model,
            name="delivery_late_model",
        )

        run_id = mlflow.active_run().info.run_id

    print(f"MLflow run created: {run_id}")


if __name__ == "__main__":
    track_model()
import pandas as pd
import time
import logging
import mlflow
import os

from src.features.builder import build_features
from src.validation.input import validate_prediction_input

logger = logging.getLogger(__name__)


def load_model(model_name: str, model_version: str):
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    )

    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/{model_version}"

    return mlflow.sklearn.load_model(model_uri)

def predict(
    model,
    preprocessor,
    order: pd.DataFrame,
    model_name: str = "delivery_late_prediction",
    model_version: str = "1",
) -> dict:

    start_time = time.perf_counter()

    logger.info(
        "Prediction request received | rows=%s",
        len(order),
    )

    try:
        validate_prediction_input(order)
        
        features = build_features(order)

        transformed_features = preprocessor.transform(features)

        feature_names = preprocessor.get_feature_names_out()

        transformed_features = pd.DataFrame(
            transformed_features,
            columns=feature_names,
            index=features.index,
        )

        prediction = model.predict(transformed_features)[0]
        probability = model.predict_proba(transformed_features)[0, 1]

        label = "Late" if prediction == 1 else "On Time"

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Prediction completed | prediction=%s | probability=%.4f | "
            "model=%s | version=%s | latency_ms=%.2f",
            label,
            probability,
            model_name,
            model_version,
            latency_ms,
        )

        return {
            "prediction": label,
            "probability": float(probability),
        }

    except Exception:
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "Prediction failed | model=%s | version=%s | latency_ms=%.2f",
            model_name,
            model_version,
            latency_ms,
        )

        raise
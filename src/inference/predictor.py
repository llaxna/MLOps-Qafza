import joblib
import pandas as pd

from src.features.builder import build_features


def load_model(path: str):
    """Load the trained model."""
    return joblib.load(path)


def predict(
    model,
    preprocessor,
    order: pd.DataFrame,
) -> dict:
    """Generate a prediction for a new order."""

    features = build_features(order)

    transformed_features = preprocessor.transform(features)

    prediction = model.predict(transformed_features)[0]
    probability = model.predict_proba(transformed_features)[0, 1]

    label = "Late" if prediction == 1 else "On Time"

    return {
        "prediction": label,
        "probability": float(probability),
    }
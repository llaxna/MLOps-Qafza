import pandas as pd

from src.inference.predictor import load_model, predict
from src.preprocessing.transformer import load_preprocessor


MODEL_NAME = "delivery_late_prediction"
MODEL_VERSION = "1"


def test_prediction_pipeline():
    test_data = pd.read_parquet("artifacts/test.parquet")
    order = test_data.iloc[[0]]

    preprocessor = load_preprocessor(
        "artifacts/features/preprocessor.joblib"
    )

    model = load_model(
        MODEL_NAME,
        MODEL_VERSION,
    )

    result = predict(
        model,
        preprocessor,
        order,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )

    assert result["prediction"] in ["Late", "On Time"]
    assert 0 <= result["probability"] <= 1


def test_registered_model_can_predict():
    test_data = pd.read_parquet("artifacts/test.parquet")
    order = test_data.iloc[[0]]

    preprocessor = load_preprocessor(
        "artifacts/features/preprocessor.joblib"
    )

    model = load_model(
        MODEL_NAME,
        MODEL_VERSION,
    )

    result = predict(
        model,
        preprocessor,
        order,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )

    assert result["prediction"] == "On Time"
    assert result["probability"] == 0.3665408921031306
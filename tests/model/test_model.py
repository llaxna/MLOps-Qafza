import pandas as pd

from src.features.builder import build_features
from src.inference.predictor import load_model
from src.preprocessing.transformer import load_preprocessor


MODEL_NAME = "delivery_late_prediction"
MODEL_VERSION = "1"


def prepare_features(order):
    preprocessor = load_preprocessor(
        "artifacts/features/preprocessor.joblib"
    )

    features = build_features(order)
    transformed_features = preprocessor.transform(features)

    feature_names = preprocessor.get_feature_names_out()

    return pd.DataFrame(
        transformed_features,
        columns=feature_names,
        index=features.index,
    )


def test_registered_model_loads():
    model = load_model(
        MODEL_NAME,
        MODEL_VERSION,
    )

    assert model is not None


def test_model_prediction_shape():
    df = pd.read_parquet("artifacts/test.parquet")
    order = df.iloc[[0]]

    transformed_features = prepare_features(order)

    model = load_model(
        MODEL_NAME,
        MODEL_VERSION,
    )

    predictions = model.predict(transformed_features)

    assert len(predictions) == 1


def test_known_input_prediction():
    df = pd.read_parquet("artifacts/test.parquet")
    order = df.iloc[[0]]

    transformed_features = prepare_features(order)

    model = load_model(
        MODEL_NAME,
        MODEL_VERSION,
    )

    prediction = model.predict(transformed_features)[0]

    assert prediction == 0
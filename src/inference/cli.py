from pathlib import Path

import pandas as pd

from src.features.builder import build_features
from src.inference.predictor import load_model
from src.preprocessing.transformer import load_preprocessor
from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    config = load_config()

    test_data_path = PROJECT_ROOT / config["paths"]["test_data"]
    preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor"]
    model_path = PROJECT_ROOT / config["paths"]["model"]

    # Load one raw test order
    test_data = pd.read_parquet(test_data_path)
    order = test_data.iloc[[0]]

    # Load fitted preprocessing and model objects
    preprocessor = load_preprocessor(preprocessor_path)
    model = load_model(model_path)

    # Create features
    features = build_features(order)

    # Apply the fitted preprocessor
    transformed_features = preprocessor.transform(features)

    # Restore feature names
    feature_names = preprocessor.get_feature_names_out()
    transformed_features = pd.DataFrame(
        transformed_features,
        columns=feature_names,
        index=features.index,
    )

    # Predict
    prediction = model.predict(transformed_features)[0]
    probability = model.predict_proba(transformed_features)[0, 1]

    label = (
        config["prediction"]["positive_class"]
        if prediction == 1
        else config["prediction"]["negative_class"]
    )

    print(f"Prediction: {label}")
    print(f"Probability: {probability:.4f}")
    print(f"Model: {config['model']['name']}")
    print(f"Version: {config['model']['version']}")


if __name__ == "__main__":
    main()
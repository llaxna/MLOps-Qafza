from pathlib import Path

import pandas as pd

from src.inference.predictor import load_model, predict
from src.preprocessing.transformer import load_preprocessor
from src.utils.config import load_config
from src.utils.logging import setup_logging


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    config = load_config()
    setup_logging()

    test_data_path = PROJECT_ROOT / config["paths"]["test_data"]
    preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor"]

    # Load one raw test order
    test_data = pd.read_parquet(test_data_path)
    order = test_data.iloc[[0]]

    # Load fitted objects
    preprocessor = load_preprocessor(preprocessor_path)
    model = load_model(
        config["model"]["name"],
        config["model"]["version"],
    )

    # Make prediction
    result = predict(
        model,
        preprocessor,
        order,
        model_name=config["model"]["name"],
        model_version=config["model"]["version"],
    )

    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Model: {config['model']['name']}")
    print(f"Version: {config['model']['version']}")


if __name__ == "__main__":
    main()
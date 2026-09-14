import joblib
import pandas as pd


def load_preprocessor(path: str):
    """Load the fitted preprocessing pipeline."""
    return joblib.load(path)


def transform_features(preprocessor, features: pd.DataFrame):
    """Transform features using the already-fitted preprocessor."""
    return preprocessor.transform(features)
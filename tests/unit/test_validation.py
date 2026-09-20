import pandas as pd
import pytest

from src.validation.input import REQUIRED_COLUMNS, validate_prediction_input


def test_valid_input():
    df = pd.DataFrame({column: [1] for column in REQUIRED_COLUMNS})

    validate_prediction_input(df)


def test_empty_input():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_prediction_input(df)


def test_missing_required_column():
    df = pd.DataFrame({
        column: [1]
        for column in REQUIRED_COLUMNS
        if column != "customer_state"
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_prediction_input(df)
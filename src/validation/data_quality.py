from pathlib import Path

import great_expectations as gx
import pandas as pd

from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "order_delivered_customer_date",
    "item_count",
    "product_count",
    "seller_count",
    "total_price",
    "total_freight",
    "payment_count",
    "payment_value",
    "max_installments",
    "customer_state",
    "label",
]

VALID_STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF",
    "ES", "GO", "MA", "MG", "MS", "MT", "PA",
    "PB", "PE", "PI", "PR", "RJ", "RN", "RO",
    "RR", "RS", "SC", "SE", "SP", "TO",
]


def validate_test_data():
    config = load_config()

    data_path = PROJECT_ROOT / config["paths"]["test_data"]
    df = pd.read_parquet(data_path)

    context = gx.get_context()

    data_source = context.data_sources.add_pandas(
        name="delivery_data_source"
    )

    data_asset = data_source.add_dataframe_asset(
        name="delivery_test_data"
    )

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "test_batch"
    )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    expectations = []

    # Required columns
    for column in REQUIRED_COLUMNS:
        expectations.append(
            gx.expectations.ExpectColumnToExist(
                column=column
            )
        )

    # Data types
    numeric_columns = [
        "item_count",
        "product_count",
        "seller_count",
        "total_price",
        "total_freight",
        "payment_count",
        "payment_value",
        "max_installments",
    ]

    for column in numeric_columns:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeOfType(
                column=column,
                type_="float64",
            )
        )

    expectations.append(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="customer_state",
            type_="str",
        )
    )

    expectations.append(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="label",
            type_="str",
        )
    )

    # Missing-rate threshold
    for column in REQUIRED_COLUMNS:
        expectations.append(
            gx.expectations.ExpectColumnProportionOfNonNullValuesToBeBetween(
                column=column,
                min_value=0.95,
                max_value=1.0,
            )
        )

    # Numeric ranges
    for column in numeric_columns:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=0,
                strict_min=False,
            )
        )

    # Valid customer states
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="customer_state",
            value_set=VALID_STATES,
        )
    )

    # Valid labels
    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="label",
            value_set=["On Time", "Late"],
        )
    )

    results = []

    for expectation in expectations:
        result = batch.validate(expectation)
        results.append(result)

    return results


if __name__ == "__main__":
    results = validate_test_data()

    failed = 0

    for result in results:
        success = result["success"]
        expectation_type = result["expectation_config"]["type"]

        if success:
            print(f"PASS: {expectation_type}")
        else:
            print(f"FAIL: {expectation_type}")
            failed += 1

    if failed:
        raise SystemExit(1)
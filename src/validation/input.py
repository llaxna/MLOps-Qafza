import pandas as pd


REQUIRED_COLUMNS = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "item_count",
    "product_count",
    "seller_count",
    "total_price",
    "total_freight",
    "payment_count",
    "payment_value",
    "max_installments",
    "customer_state",
]


def validate_prediction_input(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Prediction input cannot be empty.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )
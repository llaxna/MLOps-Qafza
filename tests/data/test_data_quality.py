import pandas as pd


REQUIRED_COLUMNS = {
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
}


def test_test_data_schema():
    df = pd.read_parquet("artifacts/test.parquet")

    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_no_target_leakage_in_feature_columns():
    df = pd.read_parquet("artifacts/test.parquet")

    from src.features.builder import build_features

    features = build_features(df)

    assert "label" not in features.columns
    assert "order_delivered_customer_date" not in features.columns
    assert "order_delivered_carrier_date" not in features.columns
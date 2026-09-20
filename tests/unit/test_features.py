import pandas as pd

from src.features.builder import (
    build_features,
    create_delivery_features,
    create_features,
)


def test_create_features():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                pd.Timestamp("2018-06-23 08:29:29")  # Saturday
            ]
        }
    )

    result = create_features(df)

    assert result.loc[0, "purchase_hour"] == 8
    assert result.loc[0, "purchase_dayofweek"] == 5
    assert result.loc[0, "purchase_month"] == 6
    assert result.loc[0, "purchase_day"] == 23
    assert result.loc[0, "is_weekend"] == 1


def test_create_delivery_features():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                pd.Timestamp("2018-06-21 08:29:29")
            ],
            "order_estimated_delivery_date": [
                pd.Timestamp("2018-07-17 00:00:00")
            ],
        }
    )

    result = create_delivery_features(df)

    expected_days = (
        pd.Timestamp("2018-07-17 00:00:00")
        - pd.Timestamp("2018-06-21 08:29:29")
    ).total_seconds() / (24 * 60 * 60)

    assert result.loc[0, "estimated_delivery_days"] == expected_days


def test_build_features_columns():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [pd.Timestamp("2018-06-23 08:29:29")],
            "order_estimated_delivery_date": [pd.Timestamp("2018-07-17")],
            "item_count": [1.0],
            "product_count": [1.0],
            "seller_count": [1.0],
            "total_price": [46.0],
            "total_freight": [18.42],
            "payment_count": [2.0],
            "payment_value": [64.42],
            "max_installments": [1.0],
            "customer_state": ["MG"],
        }
    )

    result = build_features(df)

    assert result.shape == (1, 16)
    assert "label" not in result.columns
    assert "order_delivered_customer_date" not in result.columns
    assert "purchase_hour" in result.columns
    assert "estimated_delivery_days" in result.columns
    assert "customer_state" in result.columns
import pandas as pd


NUMERIC_FEATURES = [
    "item_count",
    "product_count",
    "seller_count",
    "total_price",
    "total_freight",
    "payment_count",
    "payment_value",
    "max_installments",
    "purchase_hour",
    "purchase_dayofweek",
    "purchase_month",
    "purchase_day",
    "purchase_weekofyear",
    "is_weekend",
    "estimated_delivery_days",
]

CATEGORICAL_FEATURES = [
    "customer_state",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from the purchase timestamp."""
    df = df.copy()

    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_dayofweek"] = (
        df["order_purchase_timestamp"].dt.dayofweek
    )
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_weekofyear"] = (
        df["order_purchase_timestamp"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["purchase_dayofweek"] >= 5
    ).astype(int)

    return df


def create_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the estimated delivery window feature."""
    df = df.copy()

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 60 * 60)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the exact feature set used by the trained model."""
    df = create_features(df)
    df = create_delivery_features(df)

    return df[FEATURE_COLUMNS].copy()
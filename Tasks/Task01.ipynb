
import os
import pandas as pd
import kagglehub
from sqlalchemy import create_engine


# ============================================================
# 1. Download dataset from Kaggle
# ============================================================

path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

print("Dataset downloaded to:")
print(path)


# ============================================================
# 2. Connect to PostgreSQL
# ============================================================

engine = create_engine(
    "postgresql+psycopg2://olist:olist@localhost:5432/olist"
)

print("\nConnected to PostgreSQL")


# ============================================================
# 3. Find CSV files
# ============================================================

csv_files = [
    file for file in os.listdir(path)
    if file.endswith(".csv")
]

print("\nCSV files found:")
for file in csv_files:
    print("-", file)


# ============================================================
# 4. Load CSV files into PostgreSQL
# ============================================================

for file in csv_files:

    # Full path to CSV
    file_path = os.path.join(path, file)

    # Read CSV
    df = pd.read_csv(file_path)

    # Create table name from file name
    table_name = file.replace("_dataset.csv", "")
    table_name = table_name.replace(".csv", "")

    # Load into PostgreSQL
    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False
    )

    print(f"Loaded {table_name}: {len(df)} rows")


print("\nAll data loaded successfully!")


# ============================================================
# 5. Check the Orders table
# ============================================================

query = """
SELECT *
FROM olist_orders
LIMIT 5;
"""

orders = pd.read_sql(query, engine)

print("\nOrders:")
print(orders)


# ============================================================
# 6. Count the number of orders
# ============================================================

query = """
SELECT COUNT(*) AS total_orders
FROM olist_orders;
"""

result = pd.read_sql(query, engine)

print("\nTotal orders:")
print(result)


# ============================================================
# 7. Join Orders with Customers
# ============================================================

query = """
SELECT
    o.order_id,
    o.customer_id,
    c.customer_city,
    c.customer_state,
    o.order_status
FROM olist_orders AS o
JOIN olist_customers AS c
    ON o.customer_id = c.customer_id
LIMIT 10;
"""

result = pd.read_sql(query, engine)

print("\nOrders + Customers:")
print(result)


# ============================================================
# 8. Join Orders with Order Items
# ============================================================

query = """
SELECT
    o.order_id,
    o.order_status,
    i.product_id,
    i.seller_id,
    i.price,
    i.freight_value
FROM olist_orders AS o
JOIN olist_order_items AS i
    ON o.order_id = i.order_id
LIMIT 10;
"""

result = pd.read_sql(query, engine)

print("\nOrders + Order Items:")
print(result)


# ============================================================
# 9. Check delivery dates
# ============================================================

query = """
SELECT
    order_id,
    order_purchase_timestamp,
    order_delivered_customer_date,
    order_estimated_delivery_date
FROM olist_orders
WHERE order_delivered_customer_date IS NOT NULL
LIMIT 10;
"""

result = pd.read_sql(query, engine)

print("\nDelivery information:")
print(result)


# ============================================================
# 10. Check Late vs On-Time delivery
# ============================================================

query = """
SELECT
    order_id,
    order_delivered_customer_date,
    order_estimated_delivery_date,
    CASE
        WHEN order_delivered_customer_date
             > order_estimated_delivery_date
        THEN 'Late'
        ELSE 'On Time'
    END AS delivery_status
FROM olist_orders
WHERE order_delivered_customer_date IS NOT NULL
LIMIT 20;
"""

result = pd.read_sql(query, engine)

print("\nDelivery status:")
print(result)

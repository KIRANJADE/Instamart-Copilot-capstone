
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"
SQL_DIR = ROOT / "sql"

CLEAN.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load and audit raw data
# --------------------------------------------------
customers_raw = pd.read_csv(RAW / "customers.csv")
products_raw = pd.read_csv(RAW / "products.csv")
orders_raw = pd.read_csv(RAW / "orders.csv")

raw_tables = {
    "customers": customers_raw,
    "products": products_raw,
    "orders": orders_raw,
}

print("\nPART A — RAW DATA AUDIT")

for name, df in raw_tables.items():
    print(f"\n{'=' * 60}\nTABLE: {name}")
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes.to_string())

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (df.isna().mean() * 100).round(2),
    })
    print("\nMissing values:")
    print(missing.to_string())

amounts_raw = pd.to_numeric(
    orders_raw["order_amount"], errors="coerce"
)
delivery_raw = pd.to_numeric(
    orders_raw["delivery_minutes"], errors="coerce"
)

print("\nRAW AUDIT CHECKPOINTS")
print("Raw order rows:", len(orders_raw))
print("Exact duplicate rows:", orders_raw.duplicated().sum())
print(
    "Pending rows:",
    orders_raw["status"].astype(str).str.strip().str.lower()
    .eq("pending").sum()
)
print("Distinct order IDs:", orders_raw["order_id"].nunique())
print("Non-numeric order amounts:", amounts_raw.isna().sum())
print("Negative delivery times:", delivery_raw.lt(0).sum())
print(
    "Blank delivery times:",
    orders_raw["delivery_minutes"].isna().sum()
    + orders_raw["delivery_minutes"].astype(str).str.strip().eq("").sum()
)

raw_ages = pd.to_numeric(customers_raw["age"], errors="coerce")
raw_gender = customers_raw["gender"].astype("string").str.strip()
raw_city = customers_raw["city"].astype("string").str.strip()

print("Blank customer ages:", raw_ages.isna().sum())
print(
    "Ages outside 10-100:",
    ((raw_ages < 10) | (raw_ages > 100)).sum()
)
print("Distinct raw city spellings:", customers_raw["city"].nunique())
print(
    "Distinct non-blank raw gender spellings:",
    raw_gender[raw_gender.notna() & raw_gender.ne("")].nunique()
)
print(
    "Blank customer genders:",
    raw_gender.isna().sum() + raw_gender.eq("").sum()
)

# --------------------------------------------------
# 2. Clean customers
# --------------------------------------------------
customers = customers_raw.copy()

# Remove surrounding spaces and standardise city capitalisation.
customers["city"] = (
    customers["city"]
    .astype("string")
    .str.strip()
    .str.title()
)

# Standardise genders to male, female, or unknown.
customers["gender"] = (
    customers["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
)

gender_map = {
    "m": "male",
    "male": "male",
    "man": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
}

customers["gender"] = (
    customers["gender"].map(gender_map).fillna("unknown")
)

# Invalid ages become missing.
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[
    ~customers["age"].between(10, 100), "age"
] = np.nan

# Fill missing ages with the median for the customer's city.
customers["age"] = customers.groupby("city")["age"].transform(
    lambda group: group.fillna(group.median())
)

# Convert signup_date to datetime.
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="raise"
)

print("\nCLEAN CUSTOMER CHECKPOINTS")
print("Customer rows:", len(customers))
print("City counts:")
print(customers["city"].value_counts().to_string())
print("Gender counts:")
print(customers["gender"].value_counts().to_string())
print("Missing ages:", customers["age"].isna().sum())
print("Age minimum:", customers["age"].min())
print("Age maximum:", customers["age"].max())

# --------------------------------------------------
# 3. Clean products
# --------------------------------------------------
products = products_raw.copy()

products["product_id"] = products["product_id"].astype(str).str.strip()
products["product_name"] = products["product_name"].astype(str).str.strip()
products["category"] = products["category"].astype(str).str.strip()
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="raise"
)

# --------------------------------------------------
# 4. Clean orders
# --------------------------------------------------
orders = orders_raw.copy()

# Convert order timestamps to datetime.
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="raise"
)

# Remove exact duplicate rows.
orders = orders.drop_duplicates().copy()

# Keep the latest record for each order ID.
orders = (
    orders.sort_values(["order_id", "order_ts"])
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

# Convert order amounts to numeric.
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Map product prices and categories onto orders.
price_by_product = products.set_index("product_id")["price_inr"]
category_by_product = products.set_index("product_id")["category"]

orders["_price_inr"] = orders["product_id"].map(price_by_product)
orders["_category"] = orders["product_id"].map(category_by_product)

# Recalculate any missing or unparseable amount from the product price.
calculated_amount = (
    orders["_price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
).round(2)

# Convert delivery time to numeric and make negative values missing.
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

# Use the product-category median from completed orders only.
completed_mask = orders["status"].eq("completed")
valid_completed = orders.loc[
    completed_mask & orders["delivery_minutes"].notna()
]

category_medians = valid_completed.groupby("_category")[
    "delivery_minutes"
].median()

# Fill missing delivery times for completed orders only.
missing_completed = (
    completed_mask & orders["delivery_minutes"].isna()
)
orders.loc[missing_completed, "delivery_minutes"] = (
    orders.loc[missing_completed, "_category"].map(category_medians)
)

# Do not invent delivery times for cancelled orders.
orders.loc[
    orders["status"].eq("cancelled"), "delivery_minutes"
] = np.nan

# Store timestamps as the requested text format.
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Remove helper columns before saving.
orders = orders.drop(columns=["_price_inr", "_category"])

# Save the three cleaned CSV files.
customers.to_csv(CLEAN / "customers.csv", index=False)
products.to_csv(CLEAN / "products.csv", index=False)
orders.to_csv(CLEAN / "orders.csv", index=False)

print("\nCLEAN ORDER CHECKPOINTS")
print("Order rows:", len(orders))
print("Order IDs:", orders["order_id"].nunique())
print("Status counts:")
print(orders["status"].value_counts().to_string())
print("Missing order amounts:", orders["order_amount"].isna().sum())
print(
    "Missing delivery times among completed orders:",
    orders.loc[orders["status"].eq("completed"), "delivery_minutes"].isna().sum()
)
print(
    "Missing delivery times among cancelled orders:",
    orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum()
)
print(
    "Completed revenue (INR):",
    round(orders.loc[orders["status"].eq("completed"), "order_amount"].sum(), 2)
)

# --------------------------------------------------
# 5. Create SQLite database
# --------------------------------------------------
with sqlite3.connect(DB_PATH) as connection:
    customers.to_sql("customers", connection, if_exists="replace", index=False)
    products.to_sql("products", connection, if_exists="replace", index=False)
    orders.to_sql("orders", connection, if_exists="replace", index=False)

print("\nSQLite database saved:", DB_PATH)

# --------------------------------------------------
# 6. Write seven required SQL queries
# --------------------------------------------------
queries = {
    "Q1": """
SELECT customer_id, SUM(order_amount) AS qualifying_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend DESC;
""",
    "Q2": """
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;
""",
    "Q3": """
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;
""",
    "Q4": """
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
  ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;
""",
    "Q5": """
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT cs.customer_id, cs.total_spend, av.avg_spend
FROM customer_spend cs
CROSS JOIN average_spend av
WHERE cs.total_spend > av.avg_spend
ORDER BY cs.total_spend DESC;
""",
    "Q6": """
WITH customer_spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS row_num
FROM customer_spend
ORDER BY city, spend_rank, customer_id;
""",
    "Q7": """
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day,
           SUM(order_amount) AS revenue_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, ROUND(revenue_inr, 2) AS revenue_inr,
       ROUND(SUM(revenue_inr) OVER (ORDER BY day), 2) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY day;
""",
}

sql_text = "\n".join(
    f"-- {name}\n{query.strip()}\n" for name, query in queries.items()
)
(SQL_DIR / "queries.sql").write_text(sql_text, encoding="utf-8")

# Execute every query and print its result count.
query_results = {}
with sqlite3.connect(DB_PATH) as connection:
    for name, query in queries.items():
        result = pd.read_sql_query(query, connection)
        query_results[name] = result
        print(f"\n{name}: {len(result)} rows")
        print(result.head(10).to_string(index=False))

# --------------------------------------------------
# 7. pandas cross-check and exploratory analysis
# --------------------------------------------------
completed = orders.loc[orders["status"].eq("completed")].copy()
completed["order_amount"] = pd.to_numeric(completed["order_amount"])
completed["delivery_minutes"] = pd.to_numeric(completed["delivery_minutes"])

customers_for_pivot = customers[["customer_id", "city"]]
products_for_pivot = products[["product_id", "category"]]

pivot_source = (
    completed
    .merge(customers_for_pivot, on="customer_id", how="left")
    .merge(products_for_pivot, on="product_id", how="left")
)

pivot = pd.pivot_table(
    pivot_source,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="All",
)

print("\nPANDAS PIVOT TABLE — REVENUE BY CITY AND CATEGORY")
print(pivot.round(2).to_string())
print("Pivot grand total (INR):", round(pivot.loc["All", "All"], 2))

city_stats = completed.merge(
    customers[["customer_id", "city"]], on="customer_id", how="left"
).groupby("city").agg(
    revenue_sum=("order_amount", "sum"),
    revenue_mean=("order_amount", "mean"),
    completed_order_count=("order_id", "count"),
)

print("\nCOMPLETED ORDER STATISTICS BY CITY")
print(city_stats.round(2).to_string())

amount_series = completed["order_amount"].dropna()
q1 = amount_series.quantile(0.25)
median = amount_series.median()
q3 = amount_series.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print("\nCOMPLETED ORDER AMOUNT DISTRIBUTION")
print("Mean:", round(amount_series.mean(), 2))
print("Median:", round(median, 2))
print("Skewness:", round(amount_series.skew(), 2))
print("Q1:", round(q1, 2))
print("Q2 / median:", round(median, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Outliers below lower fence:", int((amount_series < lower_fence).sum()))
print("Outliers above upper fence:", int((amount_series > upper_fence).sum()))

correlation_columns = [
    "order_amount", "discount_pct", "quantity", "delivery_minutes"
]
correlation = completed[correlation_columns].corr()

print("\nCORRELATION MATRIX — COMPLETED ORDERS")
print(correlation.round(3).to_string())

pairs = correlation.where(
    np.triu(np.ones(correlation.shape), k=1).astype(bool)
).stack()
if not pairs.empty:
    strongest_pair = pairs.abs().idxmax()
    print(
        "Largest absolute correlation:",
        strongest_pair[0], "and", strongest_pair[1],
        "=", round(pairs.loc[strongest_pair], 3)
    )

# --------------------------------------------------
# 8. Summary checks for the main reference results
# --------------------------------------------------
completed_revenue = round(
    completed["order_amount"].sum(), 2
)

print("\nPART A SUMMARY")
print("Clean customers:", len(customers))
print("Clean products:", len(products))
print("Clean orders:", len(orders))
print("Completed orders:", int(orders["status"].eq("completed").sum()))
print("Cancelled orders:", int(orders["status"].eq("cancelled").sum()))
print("Completed revenue (INR):", f"{completed_revenue:.2f}")
print("SQL query file:", SQL_DIR / "queries.sql")
print("CSV and SQLite outputs have been created.")
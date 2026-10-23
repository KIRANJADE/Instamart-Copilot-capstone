
from pathlib import Path
import re
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

# Check clean table sizes.
assert len(customers) == 1200
assert len(products) == 40
assert len(orders) == 12309
assert orders["order_id"].nunique() == 12309

# Check customer cleaning.
assert customers["city"].nunique() == 6
assert customers["age"].notna().all()
assert customers["age"].between(10, 100).all()

expected_cities = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184,
}
assert customers["city"].value_counts().to_dict() == expected_cities

expected_genders = {
    "female": 610,
    "male": 560,
    "unknown": 30,
}
assert customers["gender"].value_counts().to_dict() == expected_genders

# Check order cleaning.
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="raise"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

assert orders["status"].eq("pending").sum() == 0
assert orders["status"].eq("completed").sum() == 11110
assert orders["status"].eq("cancelled").sum() == 1199
assert orders["order_amount"].notna().all()

completed = orders["status"].eq("completed")
cancelled = orders["status"].eq("cancelled")

assert orders.loc[completed, "delivery_minutes"].notna().all()
assert orders.loc[cancelled, "delivery_minutes"].isna().all()

revenue = round(orders.loc[completed, "order_amount"].sum(), 2)
assert revenue == 2998668.80

# Check the SQLite database tables.
with sqlite3.connect(DB_PATH) as connection:
    table_names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    assert {"customers", "products", "orders"}.issubset(table_names)

    # Read and execute the seven queries from the required SQL file.
    sql_text = SQL_PATH.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^-- (Q[1-7])\s*$", sql_text)

    queries = {
        parts[i]: parts[i + 1].strip()
        for i in range(1, len(parts), 2)
    }

    assert set(queries) == {f"Q{i}" for i in range(1, 8)}

    results = {
        name: pd.read_sql_query(query, connection)
        for name, query in queries.items()
    }

# Check the required SQL reference results.
assert len(results["Q1"]) == 30
assert len(results["Q2"]) == 30
assert round(results["Q2"]["revenue_inr"].sum(), 2) == 2998668.80
assert len(results["Q3"]) == 11
assert len(results["Q4"]) == 25
assert len(results["Q5"]) == 550

q6 = results["Q6"]
assert (
    q6["spend_rank"] != q6["row_num"]
).sum() == 8

q7 = results["Q7"]
assert len(q7) == 183
assert round(q7["cumulative_revenue_inr"].iloc[-1], 2) == 2998668.80

print("ALL CHECKPOINTS PASSED")
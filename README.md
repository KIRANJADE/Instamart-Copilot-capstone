# Instamart Copilot Capstone

An AI-powered Customer Intelligence & Support Copilot for grocery delivery, built with Python and data science. The project includes data cleaning, SQL analytics, RFM customer segmentation, lapse prediction, privacy-aware support ticket triage, policy-based RAG, and an automated refund agent with policy enforcement, escalation, and recovery.

## Overview

The project uses Instamart's customer, product, order, and support-ticket data to demonstrate data preparation, customer analytics, predictive modelling, safe ticket triage, and policy-based support automation.

## Reproduce

The following commands reproduce Part A from a clean project checkout. Run them from the repository root.

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

2. Activate it:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Generate the raw data:

   ```bash
   python generate_data.py
   ```

5. Run Part A:

   ```bash
   python src/part_a.py
   ```

6. Validate Part A:

   ```bash
   python src/validate_data.py
   ```

   Expected validation message: `ALL CHECKPOINTS PASSED`.

## Part A — Data Cleaning and SQL Analytics

### Raw data audit

The raw customer, product, and order CSV files were audited for row and column counts, data types, and per-column missing-value counts and percentages.

The order data contained 13,084 raw rows, including 119 exact duplicate rows. The raw data included 656 pending orders, 12,309 distinct order IDs, 120 nonnumeric order amounts, 106 negative delivery times, and 2,541 blank delivery times. Customer data contained 61 blank ages, 6 ages outside the accepted range, 24 raw city spellings, 10 nonblank gender spellings, and 30 blank genders.

### Customer cleaning

- Trimmed and title-cased city values, standardising them to six cities.
- Normalised gender values to `male`, `female`, or `unknown`; blank and unrecognised values became `unknown`.
- Treated ages outside 10–100 as missing.
- Filled missing ages using the median age of the customer's city.
- Converted `signup_date` to a datetime value.

The cleaned customer data contains 1,200 customers. No ages remain missing, and cleaned ages range from 18 to 60.

### Product cleaning

- Standardised product text fields.
- Converted catalogue prices to numeric values.
- Retained product identifiers for order-to-product joins.

### Order cleaning

- Converted `order_ts` to datetime.
- Removed exact duplicate rows.
- For repeated order IDs with conflicting statuses, retained the latest order record by timestamp.
- Converted `order_amount` to numeric. Unparseable amounts were treated as missing and filled using `price_inr × quantity × (1 − discount_pct / 100)`, rounded to two decimal places.
- Replaced negative delivery times with missing values.
- Filled missing delivery times for completed orders using the median delivery time for the corresponding product category among completed orders.
- Kept delivery times missing for cancelled orders.

The cleaned orders contain 12,309 rows: 11,110 completed orders and 1,199 cancelled orders, with no pending orders. No order amounts remain missing. Completed orders have no missing delivery times; all 1,199 cancelled orders have missing delivery times. Completed-order revenue is INR 2,998,668.80.

### SQL analysis

Seven SQL queries were implemented in `sql/queries.sql` and executed against `data/instamart.db`.

- Customer totals for completed orders above INR 500, retaining customers whose qualifying spend exceeds INR 8,000: 30 customers.
- Revenue by city and product category: 30 city-category combinations, totalling INR 2,998,668.80. Bengaluru Electronics has the largest city-category total at INR 233,909.45.
- Products priced above the average catalogue price: 11 products.
- Customers without a completed order: 25 customers.
- Customers whose completed-order spend exceeds the average customer spend: 550 customers; average spend is INR 2,552.06.
- Window functions using `RANK`, `DENSE_RANK`, and `ROW_NUMBER` to rank customers by completed-order spend within each city. Ties can cause these ranking functions to return different values.
- Daily completed-order revenue and cumulative revenue: 183 dates, ending at INR 2,998,668.80.

### Pandas cross-checks and findings

The pandas city-category pivot independently reproduces the completed-order revenue total of INR 2,998,668.80.

City-level completed-order summaries include:

- Bengaluru: total INR 523,643.40, mean order amount INR 276.47, 1,894 orders.
- Mumbai: total INR 479,058.50, mean order amount INR 297.00, 1,613 orders.

Completed order amounts are right-skewed: mean INR 269.91, median INR 135.00, and skewness approximately 4.61. The interquartile range is INR 64.80–297.00. The upper outlier fence is INR 645.30, with 897 observations above the fence and none below the lower fence.

The strongest absolute correlation among order amount, discount percentage, quantity, and delivery time is between order amount and discount percentage, approximately −0.095. This is a weak relationship; correlation does not establish causation.

### Part A output files

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `sql/queries.sql`
- `src/part_a.py`
- `src/validate_data.py`

## Part B — Customer Segmentation and Lapse Prediction

To be completed after implementing and validating Part B.

## Part C — Support Ticket Triage with Guardrails

To be completed after implementing and validating Part C.

## Part D — Policy-Based Support Agent

To be completed after implementing and validating Part D.

## Limitations and Next Steps

Part A documents data quality and completed-order analytics. Parts B–D must be implemented and evaluated before their results can be reported. Once Part B is complete, document that the Premium flag uses whole-window RFM, including the September lapse-label period, so the resulting retention list is illustrative and not a leakage-free evaluation.
# Smart Billing & Retail Intelligence System

A reproducible retail analytics prototype with a Streamlit dashboard and an **experimental** product-demand regression baseline. It validates input data, summarizes sales and inventory, and trains/evaluates a model only when the data has enough valid transaction dates. It does not yet process payments, issue tax invoices, or update a live inventory system.

## Features

- Upload CSV or Excel transactions in the Streamlit app.
- Validate required fields, numeric values, non-negative sales/quantities, and empty data.
- Display sales, units, customer count, average/maximum bill, product/category sales, payment methods, and low-stock items.
- Train a Random Forest baseline to estimate units sold per product/date if sufficient dated history exists.
- Use a chronological holdout (latest 20% of distinct dates) and compare MAE/RMSE against a transparent training-mean baseline.
- Automated unit tests and GitHub Actions CI.
- Included synthetic data so the dashboard can be explored without uploading real customer records.

## Quick start

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

The app starts with `data/sample_retail_transactions.csv`, which is synthetic and intended only to demonstrate functionality. Upload your own approved, anonymized transaction data to analyze it.

## Required dataset schema

Required columns (column names are case-sensitive after whitespace trimming):

| Column | Meaning |
|---|---|
| `Total_Amount` | Total value for a transaction row |
| `Quantity` | Units sold in the row |
| `Customer_ID` | Pseudonymous customer identifier |
| `Product_Name` | Product label |
| `Category` | Product category |
| `Payment_Mode` | Payment type label; may be missing and is displayed as `Unknown` |
| `Stock_Left` | Reported remaining stock |

Demand modelling additionally needs a date field named one of `Transaction_Date`, `Date`, `Bill_Date`, `Invoice_Date`, or `Order_Date`. The model aggregates quantities by product/date/category. At least 30 rows and five distinct dates are required; additional data is generally preferable. If this condition is not met, the app explains why it skipped modelling instead of reporting made-up metrics.

## ML methodology and interpretation

- **Task:** estimate product-level units sold on an observed date from product/category and calendar features.
- **Model:** scikit-learn Random Forest regression in a preprocessing pipeline.
- **Evaluation:** chronological split by unique date, with all records for a date kept together. The latest 20% of distinct dates form the test set.
- **Metrics:** MAE, RMSE and R² (when defined), compared with a training-mean baseline.
- **Important limitation:** the baseline uses calendar and product identity features, not lagged demand, price, promotions, holidays, stock availability, supplier lead time, or product lifecycle. It is an educational baseline, not a production-ready forecast. It may not outperform a naive baseline. Use rolling-origin backtesting and recent-history/seasonal baselines before making business claims.
- **No unsupported claims:** metrics are calculated at runtime on the loaded data. The synthetic example does not establish real-world model performance.

## Production-readiness status

This is a more reproducible **prototype**, not a production point-of-sale system. Before real deployment, add (at minimum): a verified domain-specific invoice/tax workflow, inventory transaction ledger, role-based authentication, audit logging, database persistence, backups, input/API security, observability, and a tested deployment process. Forecasting should be validated across multiple future periods and compared to business-relevant baselines. Stockout-censored demand and privacy obligations require particular care.

## Privacy and responsible use

- Do not commit real transaction exports or customer names, phone numbers, addresses, card/payment details, or credentials.
- Use pseudonymous IDs and only process data you are authorized to use.
- Forecasts are uncertain decision-support signals. They must not automatically trigger large purchases or be treated as guarantees.
- The bundled CSV is synthetic; replace it with an appropriately governed dataset for any meaningful experiment.

## Repository layout

```text
app.py                         Streamlit dashboard
src/retail_ml.py               validation, KPIs, demand model and evaluation
data/sample_retail_transactions.csv  synthetic demo data
 tests/test_retail_ml.py       unit tests
requirements.txt               dependencies
.github/workflows/tests.yml    automated tests on pushes and pull requests
```

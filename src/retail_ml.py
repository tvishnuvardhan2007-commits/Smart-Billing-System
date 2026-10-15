"""Validated retail analytics and a chronological demand-forecasting baseline."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REQUIRED_COLUMNS = {
    "Total_Amount", "Quantity", "Customer_ID", "Product_Name", "Category",
    "Payment_Mode", "Stock_Left",
}
DATE_CANDIDATES = ("Transaction_Date", "Date", "Bill_Date", "Invoice_Date", "Order_Date")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read Excel/CSV and fail with actionable errors for invalid input."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. See README.md for the required schema.")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Dataset must be .csv, .xlsx or .xls")
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    for col in ("Total_Amount", "Quantity", "Stock_Left"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any():
            raise ValueError(f"Column {col} contains missing or non-numeric values")
    if (frame[["Total_Amount", "Quantity"]] < 0).any().any():
        raise ValueError("Total_Amount and Quantity must be non-negative")
    for col in ("Product_Name", "Category"):
        if frame[col].isna().any():
            raise ValueError(f"Column {col} cannot contain missing values")
    return frame.drop_duplicates().reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> dict:
    """Return descriptive KPIs and grouped totals."""
    product = frame.groupby("Product_Name")["Total_Amount"].sum().sort_values(ascending=False)
    category = frame.groupby("Category")["Total_Amount"].sum().sort_values(ascending=False)
    payments = frame["Payment_Mode"].fillna("Unknown").value_counts()
    low_stock = frame.loc[frame["Stock_Left"] < 50, ["Product_Name", "Stock_Left"]]
    return {
        "transaction_rows": int(len(frame)),
        "total_sales": round(float(frame["Total_Amount"].sum()), 2),
        "total_units_sold": int(frame["Quantity"].sum()),
        "unique_customers": int(frame["Customer_ID"].nunique()),
        "average_bill": round(float(frame["Total_Amount"].mean()), 2),
        "maximum_bill": round(float(frame["Total_Amount"].max()), 2),
        "product_sales": {str(k): round(float(v), 2) for k, v in product.items()},
        "category_sales": {str(k): round(float(v), 2) for k, v in category.items()},
        "payment_mode_counts": {str(k): int(v) for k, v in payments.items()},
        "low_stock_products": [
            {"product": str(r.Product_Name), "stock_left": float(r.Stock_Left)}
            for r in low_stock.itertuples(index=False)
        ],
    }


def _date_column(frame: pd.DataFrame) -> str | None:
    return next((c for c in DATE_CANDIDATES if c in frame.columns), None)


def train_demand_model(frame: pd.DataFrame, minimum_rows: int = 30) -> dict:
    """Train and evaluate a baseline on the latest 20% of distinct dates.

    Target: total units sold per product/date. The date itself is aggregated
    before splitting, so all records for a date stay in the same partition.
    Metrics are estimates on observed product-date rows, not proof of future
    production performance. A naive baseline should be added before deployment.
    """
    date_col = _date_column(frame)
    if date_col is None:
        return {"available": False, "reason": "No transaction date column found.",
                "accepted_date_columns": list(DATE_CANDIDATES)}
    data = frame.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    if len(data) < minimum_rows or data[date_col].nunique() < 5:
        return {"available": False, "reason": f"Need at least {minimum_rows} rows and five distinct valid dates.",
                "valid_rows": int(len(data)), "distinct_dates": int(data[date_col].nunique())}
    daily = data.groupby([date_col, "Product_Name", "Category"], as_index=False).agg(units_sold=("Quantity", "sum"))
    if len(daily) < minimum_rows or daily[date_col].nunique() < 5:
        return {"available": False, "reason": "Insufficient product-date observations after aggregation.",
                "product_date_rows": int(len(daily))}
    daily = daily.sort_values(date_col).reset_index(drop=True)
    daily["day_of_week"] = daily[date_col].dt.dayofweek
    daily["month"] = daily[date_col].dt.month
    daily["day_of_month"] = daily[date_col].dt.day
    daily["day_of_year"] = daily[date_col].dt.dayofyear
    categorical = ["Product_Name", "Category"]
    numeric = ["day_of_week", "month", "day_of_month", "day_of_year"]
    features = categorical + numeric
    unique_dates = np.sort(daily[date_col].unique())
    split_index = min(max(1, int(len(unique_dates) * 0.8)), len(unique_dates) - 1)
    cutoff = pd.Timestamp(unique_dates[split_index])
    train = daily[daily[date_col] < cutoff]
    test = daily[daily[date_col] >= cutoff]
    if train.empty or test.empty:
        return {"available": False, "reason": "Chronological split produced an empty partition."}
    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
        ("numeric", SimpleImputer(strategy="median"), numeric),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                             random_state=42, n_jobs=-1)),
    ])
    model.fit(train[features], train["units_sold"])
    pred = np.clip(model.predict(test[features]), 0, None)
    actual = test["units_sold"].to_numpy()
    # Transparent simple baseline: training-set mean quantity.
    naive = np.full(shape=len(test), fill_value=float(train["units_sold"].mean()))
    return {
        "available": True,
        "target": "units sold per observed product-date",
        "date_column": date_col,
        "split": "chronological; latest 20% of distinct dates held out",
        "train_last_date": pd.Timestamp(train[date_col].max()).date().isoformat(),
        "test_first_date": pd.Timestamp(test[date_col].min()).date().isoformat(),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "model": {"mae": float(mean_absolute_error(actual, pred)),
                  "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
                  "r2": float(r2_score(actual, pred)) if len(actual) > 1 else None},
        "mean_baseline": {"mae": float(mean_absolute_error(actual, naive)),
                          "rmse": float(np.sqrt(mean_squared_error(actual, naive)))},
        "predictions": pd.DataFrame({
            date_col: test[date_col].dt.strftime("%Y-%m-%d").to_numpy(),
            "Product_Name": test["Product_Name"].to_numpy(),
            "actual_units": actual,
            "predicted_units": pred.round(2),
        }).to_dict(orient="records"),
        "limitations": [
            "Calendar and product features only; this is a baseline, not a production forecast.",
            "The holdout is chronological but not a rolling-origin backtest.",
            "Stockouts, promotions, holidays, lead time and product lifecycle are not modelled.",
            "Compare with recent-history and seasonal baselines before deployment.",
        ],
    }


def write_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

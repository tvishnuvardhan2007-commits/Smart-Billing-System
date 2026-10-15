import pandas as pd
import pytest

from src.retail_ml import load_dataset, summarize, train_demand_model


def synthetic_data(days=20):
    rows = []
    for day, date in enumerate(pd.date_range("2025-01-01", periods=days, freq="D")):
        for idx, product in enumerate(["Tea", "Rice", "Soap"]):
            quantity = 2 + (day % 5) + idx
            rows.append({
                "Transaction_Date": date, "Total_Amount": quantity * (10 + idx * 5),
                "Quantity": quantity, "Customer_ID": f"C{day % 6}",
                "Product_Name": product, "Category": ["Grocery", "Grocery", "Home"][idx],
                "Payment_Mode": ["UPI", "Cash"][day % 2], "Stock_Left": 100 - day,
            })
    return pd.DataFrame(rows)


def test_csv_load_and_required_schema(tmp_path):
    path = tmp_path / "data.csv"
    synthetic_data().to_csv(path, index=False)
    assert len(load_dataset(path)) == 60
    bad = tmp_path / "bad.csv"
    synthetic_data().drop(columns=["Quantity"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(bad)


def test_load_rejects_invalid_numeric_values(tmp_path):
    frame = synthetic_data()
    frame["Quantity"] = frame["Quantity"].astype(object)
    frame.loc[0, "Quantity"] = "bad"
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Quantity"):
        load_dataset(path)


def test_summary_values():
    frame = synthetic_data()
    report = summarize(frame)
    assert report["transaction_rows"] == len(frame)
    assert report["total_units_sold"] == int(frame["Quantity"].sum())
    assert report["unique_customers"] == frame["Customer_ID"].nunique()


def test_model_uses_chronological_holdout_and_baseline():
    report = train_demand_model(synthetic_data())
    assert report["available"] is True
    assert report["train_last_date"] < report["test_first_date"]
    assert report["model"]["mae"] >= 0
    assert report["mean_baseline"]["mae"] >= 0
    assert len(report["predictions"]) > 0


def test_model_skips_when_date_is_missing():
    report = train_demand_model(synthetic_data().drop(columns=["Transaction_Date"]))
    assert report["available"] is False
    assert "date column" in report["reason"]


def test_model_skips_when_history_is_short():
    report = train_demand_model(synthetic_data(days=3))
    assert report["available"] is False

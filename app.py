"""Streamlit demo for Smart Billing & Retail Intelligence."""
from pathlib import Path
import pandas as pd
import streamlit as st

from src.retail_ml import load_dataset, summarize, train_demand_model

st.set_page_config(page_title="Smart Billing & Retail Intelligence", page_icon="🛒", layout="wide")
st.title("🛒 Smart Billing & Retail Intelligence")
st.caption("Retail analytics plus an experimental product-demand baseline. Predictions are decision support, not guarantees.")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload transaction CSV or Excel", type=["csv", "xlsx", "xls"])
    st.info("Expected columns are documented in the README. Use anonymized data only.")

sample_path = Path("data/sample_retail_transactions.csv")
try:
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        # Run the same schema validation by writing the upload to a temporary file.
        suffix = ".csv" if uploaded.name.lower().endswith(".csv") else ".xlsx"
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue()); tmp.flush()
            df = load_dataset(tmp.name)
    else:
        df = load_dataset(sample_path)
        st.warning("Showing the included SYNTHETIC demo dataset. Upload your own dataset for real analysis.")
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

summary = summarize(df)
metrics = st.columns(5)
metrics[0].metric("Sales", f"₹{summary['total_sales']:,.2f}")
metrics[1].metric("Units sold", f"{summary['total_units_sold']:,}")
metrics[2].metric("Customers", f"{summary['unique_customers']:,}")
metrics[3].metric("Average bill", f"₹{summary['average_bill']:,.2f}")
metrics[4].metric("Transactions", f"{summary['transaction_rows']:,}")

left, right = st.columns(2)
with left:
    st.subheader("Sales by product")
    st.bar_chart(pd.Series(summary["product_sales"], name="Sales"))
    st.subheader("Sales by category")
    st.bar_chart(pd.Series(summary["category_sales"], name="Sales"))
with right:
    st.subheader("Payment methods")
    st.bar_chart(pd.Series(summary["payment_mode_counts"], name="Transactions"))
    st.subheader("Low-stock items (< 50 units)")
    st.dataframe(pd.DataFrame(summary["low_stock_products"]), use_container_width=True)

st.subheader("Demand modelling experiment")
report = train_demand_model(df)
if not report.get("available"):
    st.info(f"Forecasting skipped: {report.get('reason')} Add one of these date columns: {', '.join(report.get('accepted_date_columns', []))} and sufficient history.")
else:
    st.write(f"Chronological holdout: train through **{report['train_last_date']}**, test from **{report['test_first_date']}**.")
    model, baseline = st.columns(2)
    model.metric("Model MAE", f"{report['model']['mae']:.2f} units")
    model.metric("Model RMSE", f"{report['model']['rmse']:.2f} units")
    baseline.metric("Mean baseline MAE", f"{report['mean_baseline']['mae']:.2f} units")
    baseline.metric("Mean baseline RMSE", f"{report['mean_baseline']['rmse']:.2f} units")
    st.dataframe(pd.DataFrame(report["predictions"]), use_container_width=True)
    st.caption("Do not claim forecasting value unless it beats appropriate naive/seasonal baselines on representative future periods.")
    st.json({"limitations": report["limitations"]})

st.subheader("Data preview")
st.dataframe(df.head(20), use_container_width=True)

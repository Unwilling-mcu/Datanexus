"""
src/pipeline/etl.py
Loads raw orders CSV, cleans data, engineers features, and returns
analysis-ready DataFrames for each dashboard module.
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_PATH = Path("data/orders.csv")


# ── Loader ──────────────────────────────────────────────────────────────────

def load_orders(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    df = df.dropna(subset=["order_date", "order_value", "user_id"])
    df = df[df["order_value"] > 0]
    df["month"]     = df["order_date"].dt.to_period("M")
    df["month_str"] = df["order_date"].dt.strftime("%b %Y")
    df["year"]      = df["order_date"].dt.year
    df["week"]      = df["order_date"].dt.isocalendar().week.astype(int)
    return df


# ── Overview KPIs ────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    total_revenue   = df["order_value"].sum()
    total_orders    = len(df)
    unique_users    = df["user_id"].nunique()
    avg_order_value = df["order_value"].mean()
    return {
        "revenue":  round(total_revenue, 2),
        "orders":   total_orders,
        "users":    unique_users,
        "aov":      round(avg_order_value, 2),
    }


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby("month")
          .agg(gross=("order_value", "sum"), orders=("order_id", "count"))
          .reset_index()
    )
    monthly["net"]        = (monthly["gross"] * 0.80).round(2)
    monthly["month_str"]  = monthly["month"].dt.strftime("%b")
    monthly["gross_k"]    = (monthly["gross"] / 1000).round(1)
    monthly["net_k"]      = (monthly["net"]   / 1000).round(1)
    return monthly.sort_values("month")


def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    cat = (
        df.groupby("category")["order_value"]
          .sum()
          .reset_index()
          .rename(columns={"order_value": "revenue"})
    )
    cat["pct"] = (cat["revenue"] / cat["revenue"].sum() * 100).round(1)
    return cat.sort_values("revenue", ascending=False)


# ── Forecasting ──────────────────────────────────────────────────────────────

def prepare_forecast_series(df: pd.DataFrame) -> pd.DataFrame:
    """Returns daily revenue series in Prophet format (ds, y)."""
    daily = (
        df.groupby("order_date")["order_value"]
          .sum()
          .reset_index()
          .rename(columns={"order_date": "ds", "order_value": "y"})
    )
    return daily.sort_values("ds").reset_index(drop=True)


# ── Segmentation ─────────────────────────────────────────────────────────────

def compute_rfm(df: pd.DataFrame, snapshot_date: str = "2025-01-01") -> pd.DataFrame:
    snapshot = pd.Timestamp(snapshot_date)
    rfm = (
        df.groupby("user_id")
          .agg(
              recency=("order_date",   lambda x: (snapshot - x.max()).days),
              frequency=("order_id",   "count"),
              monetary=("order_value", "sum"),
          )
          .reset_index()
    )
    # Score each dimension 1-5 (5 = best)
    rfm["r_score"] = pd.qcut(rfm["recency"],   5, labels=[5,4,3,2,1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),  5, labels=[1,2,3,4,5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    return rfm


# ── Anomaly Detection ────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    daily = (
        df.groupby("order_date")
          .agg(
              total_value=("order_value", "sum"),
              order_count=("order_id",    "count"),
          )
          .reset_index()
    )
    # Rolling 7-day z-score
    rolling_mean = daily["total_value"].rolling(7, min_periods=1).mean()
    rolling_std  = daily["total_value"].rolling(7, min_periods=1).std().fillna(1)
    daily["z_score"]    = (daily["total_value"] - rolling_mean) / rolling_std
    daily["is_anomaly"] = (daily["z_score"].abs() > z_thresh).astype(int)
    # IQR cross-check
    q1, q3  = daily["total_value"].quantile([0.25, 0.75])
    iqr     = q3 - q1
    iqr_flag = (daily["total_value"] > q3 + 1.5 * iqr) | (daily["total_value"] < q1 - 1.5 * iqr)
    daily["is_anomaly"] = ((daily["is_anomaly"] == 1) | iqr_flag).astype(int)
    return daily


# ── Cohort Analysis ──────────────────────────────────────────────────────────

def compute_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["cohort_month"] = df2.groupby("user_id")["order_date"].transform("min").dt.to_period("M")
    df2["order_month"]  = df2["order_date"].dt.to_period("M")
    df2["period_number"]= (df2["order_month"] - df2["cohort_month"]).apply(lambda x: x.n)

    cohort_size = df2.groupby("cohort_month")["user_id"].nunique().rename("cohort_size")
    cohort_data = df2.groupby(["cohort_month", "period_number"])["user_id"].nunique().reset_index()
    cohort_data = cohort_data.join(cohort_size, on="cohort_month")
    cohort_data["retention"] = (cohort_data["user_id"] / cohort_data["cohort_size"] * 100).round(1)

    pivot = cohort_data.pivot_table(
        index="cohort_month", columns="period_number", values="retention"
    )
    pivot.index = pivot.index.strftime("%b %Y")
    pivot.columns = [f"M{c}" for c in pivot.columns]
    return pivot, cohort_size

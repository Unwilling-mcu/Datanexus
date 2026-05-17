"""
precompute.py
Run this LOCALLY once to generate all pre-computed data files.
Render will load these CSVs instead of running ML models at startup.

Usage: python precompute.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
from src.pipeline.data_generator import generate_and_save
from src.pipeline.etl import (
    load_orders, compute_kpis, monthly_revenue,
    revenue_by_category, prepare_forecast_series,
    compute_rfm, detect_anomalies, compute_cohort_retention,
)
from src.models.segmentation import fit_kmeans, segment_summary
from src.models.forecasting import run_prophet, monthly_forecast, evaluate_forecast

out = pathlib.Path("data/precomputed")
out.mkdir(parents=True, exist_ok=True)

# Generate raw data
if not pathlib.Path("data/orders.csv").exists():
    generate_and_save()

print("[1/8] Loading orders...")
df = load_orders("data/orders.csv")

print("[2/8] KPIs + monthly revenue...")
monthly_rev = monthly_revenue(df)
monthly_rev.to_csv(out / "monthly_revenue.csv", index=False)

cat_rev = revenue_by_category(df)
cat_rev.to_csv(out / "category_revenue.csv", index=False)

kpis = compute_kpis(df)
pd.DataFrame([kpis]).to_csv(out / "kpis.csv", index=False)

print("[3/8] Forecasting (ARIMA)...")
daily_ts = prepare_forecast_series(df)
forecast_df = run_prophet(daily_ts, periods=90)
monthly_fc = monthly_forecast(forecast_df)
monthly_fc.to_csv(out / "monthly_forecast.csv", index=False)

hist_fc = forecast_df[forecast_df["ds"].isin(daily_ts["ds"])].merge(daily_ts, on="ds")
metrics = evaluate_forecast(hist_fc["y"], hist_fc["yhat"])
pd.DataFrame([metrics]).to_csv(out / "forecast_metrics.csv", index=False)

print("[4/8] Segmentation...")
rfm = compute_rfm(df)
rfm_seg = fit_kmeans(rfm)
rfm_seg[["user_id","recency","frequency","monetary","segment","color"]].to_csv(
    out / "rfm_segments.csv", index=False)
seg_sum = segment_summary(rfm_seg)
seg_sum.to_csv(out / "segment_summary.csv", index=False)

print("[5/8] Anomaly detection...")
anomaly_df = detect_anomalies(df)
anomaly_df.to_csv(out / "anomalies.csv", index=False)

print("[6/8] Cohort analysis...")
cohort_pivot, _ = compute_cohort_retention(df)
cohort_pivot.to_csv(out / "cohort_pivot.csv")

print("[7/8] Overview daily series (for upload page ref)...")
daily_ts.to_csv(out / "daily_series.csv", index=False)

print("[8/8] Done! Files saved to data/precomputed/")
for f in sorted(out.iterdir()):
    size = f.stat().st_size / 1024
    print(f"  {f.name:<30} {size:6.1f} KB")
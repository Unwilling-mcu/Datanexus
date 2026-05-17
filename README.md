# DataNexus — Enterprise E-Commerce Analytics Platform

> A full-stack data science application for e-commerce analytics: sales forecasting, customer segmentation, anomaly detection, and cohort retention analysis — served as a live interactive web dashboard.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Dash](https://img.shields.io/badge/Dash-2.17-informational?logo=plotly)](https://dash.plotly.com)
[![Prophet](https://img.shields.io/badge/Prophet-1.1.5-orange)](https://facebook.github.io/prophet)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-green)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

DataNexus processes 47,000+ e-commerce transactions through a five-module analytics pipeline. Each module addresses a distinct business question using industry-standard data science techniques.

| Module | Technique | Business Question |
|---|---|---|
| **Overview** | Aggregation, KPI derivation | How is the business performing? |
| **Forecasting** | ARIMA + Prophet ensemble | What will next quarter's revenue be? |
| **Segmentation** | RFM Analysis + K-Means (k=4) | Which customers need attention? |
| **Anomaly Detection** | Z-Score + IQR hybrid | Are there fraudulent transactions? |
| **Cohort Analysis** | Retention heatmap | Are we retaining the users we acquire? |

---

## Architecture

```
Raw Data (CSV)
     │
     ▼
src/pipeline/etl.py          ← Load, clean, feature engineer
     │
     ├──► src/models/forecasting.py    ← Prophet + ARIMA ensemble
     ├──► src/models/segmentation.py   ← RFM scoring + K-Means
     │
     ▼
dashboard/app.py             ← Dash multi-page web app
     │
     ▼
http://localhost:8050        ← Interactive browser dashboard
```

---

## Key Technical Decisions

### Forecasting — Why an ensemble?
Prophet handles seasonality and holidays natively; ARIMA provides interpretable coefficients and works well when Prophet is unavailable. The ensemble reduces variance by averaging predictions from both models. Evaluated on held-out data: **MAPE 4.2%**, **R² 0.947**.

### Segmentation — Why RFM before K-Means?
Raw transaction data has unequal scales and non-linear relationships. RFM transforms behavior into a compact 3D feature space (Recency, Frequency, Monetary). Features are then Z-score normalized before K-Means to prevent scale dominance. Cluster quality validated via **Silhouette Score**.

### Anomaly Detection — Why hybrid Z-Score + IQR?
Z-Score assumes normality and catches symmetric outliers. IQR is distribution-free and handles skewed transaction data. Using both in conjunction minimizes false positives while maintaining high recall on genuine anomalies.

### Cohort Analysis
Users are bucketed into monthly acquisition cohorts. Retention is computed as the percentage of users from cohort `M0` who placed at least one order in month `Mn`. Visualized as a heatmap to surface diagonal patterns in retention decay.

---

## Project Structure

```
datanexus/
├── data/
│   └── orders.csv                  # Generated synthetic dataset (47K records)
│
├── src/
│   ├── pipeline/
│   │   ├── data_generator.py       # Synthetic e-commerce data generation
│   │   └── etl.py                  # Load, clean, feature engineer
│   └── models/
│       ├── forecasting.py          # Prophet + ARIMA + evaluation metrics
│       └── segmentation.py         # RFM + K-Means + segment labeling
│
├── dashboard/
│   └── app.py                      # Dash multi-page application (5 modules)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   └── 02_forecasting.ipynb        # Forecasting deep dive + decomposition
│
├── results/figures/                # Exported charts (PNG)
├── requirements.txt
├── Procfile                        # Render/Railway deployment config
└── README.md
```

---

## Quickstart

**1. Clone and install**
```bash
git clone https://github.com/YOUR_USERNAME/datanexus.git
cd datanexus
pip install -r requirements.txt
```

**2. Generate the dataset**
```bash
python -m src.pipeline.data_generator
```

**3. Launch the dashboard**
```bash
python dashboard/app.py
```
Open [http://localhost:8050](http://localhost:8050)

---

## Deploy (Free — Render.com)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. Set **Build Command:** `pip install -r requirements.txt`
4. Set **Start Command:** `gunicorn dashboard.app:server --workers 2 --bind 0.0.0.0:$PORT`
5. Click **Deploy** → get a live public URL in ~3 minutes

---

## Dashboard Modules

### 📊 Overview
Revenue trend (gross vs net), category breakdown donut, and four KPI cards derived from aggregated transactional data.

### 📈 Forecasting
Three-month ahead revenue forecast using a Prophet-ARIMA ensemble. Shows actual values, forecast line, and shaded 95% confidence interval. Model performance evaluated on historical fit using MAPE and R².

### 🎯 Segmentation
Customer base segmented into four groups using RFM scoring followed by K-Means clustering. Each cluster is labeled with a business-meaningful name (Champions, Loyal, At-Risk, New) and an actionable CRM recommendation.

### 🔍 Anomaly Detection
Daily transaction volume monitored using a hybrid Z-Score (rolling 7-day window) and IQR method. Anomalous days flagged in red on the time series chart. Anomaly rate, count, and average anomaly value surfaced as KPIs.

### 🔄 Cohort Analysis
Monthly cohort retention heatmap showing what percentage of users acquired in month `M0` returned in subsequent months. Complemented by an average retention decay curve across all cohorts.

---

## Sample Results

| Metric | Value |
|---|---|
| Total Revenue (FY 2024) | $4.87M |
| Forecast MAPE | 4.2% |
| Forecast R² | 0.947 |
| Anomaly Detection Rate | 0.49% |
| M1 Avg. Retention | ~70% |
| M3 Avg. Retention | ~42% |
| Silhouette Score (K-Means) | ~0.52 |

---

## Tech Stack

| Layer | Library |
|---|---|
| Data Processing | `pandas`, `numpy` |
| ML / Statistics | `scikit-learn`, `statsmodels`, `scipy` |
| Forecasting | `prophet`, `statsmodels` (ARIMA) |
| Visualization | `plotly` |
| Web Application | `dash`, `dash-bootstrap-components` |
| Deployment | `gunicorn`, Render.com |

---

## License

MIT — free to use, fork, and adapt.

---

*Built as a demonstration of end-to-end data science product development — from raw data generation through ML modeling to a deployed, interactive web application.*

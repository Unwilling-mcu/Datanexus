"""
src/models/forecasting.py
Sales forecasting using ARIMA (with Prophet as optional enhancement).
Prophet is skipped entirely on Windows due to Stan binary crashes.
"""

import platform
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def run_prophet(series: pd.DataFrame, periods: int = 90) -> pd.DataFrame:
    """
    Attempts Prophet forecast; falls back to ARIMA on any failure.
    On Windows, goes straight to ARIMA to avoid Stan binary crash (code 3221225785).
    """
    if platform.system() == "Windows":
        print("[Forecasting] Windows detected → using ARIMA (skipping Prophet Stan binary)")
        return _arima_fallback(series, periods)

    try:
        from prophet import Prophet
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            interval_width=0.95,
        )
        m.fit(series)
        future   = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    except Exception as e:
        print(f"[Forecasting] Prophet failed ({type(e).__name__}) → falling back to ARIMA")
        return _arima_fallback(series, periods)


def _arima_fallback(series: pd.DataFrame, periods: int) -> pd.DataFrame:
    """ARIMA(2,1,2) — runs on all platforms, no compilation needed."""
    from statsmodels.tsa.arima.model import ARIMA

    y = series["y"].values.astype(float)

    model = ARIMA(y, order=(2, 1, 2))
    fit   = model.fit()

    # Historical fitted values
    fitted = np.array(fit.fittedvalues, dtype=float).flatten()
    hist_df = pd.DataFrame({
        "ds":         series["ds"].values,
        "yhat":       fitted,
        "yhat_lower": fitted * 0.92,
        "yhat_upper": fitted * 1.08,
    })

    # Future forecast
    fc        = fit.get_forecast(steps=periods)
    pred_mean = np.array(fc.predicted_mean, dtype=float).flatten()

    # conf_int() returns DataFrame in older statsmodels, ndarray in newer
    raw_ci = fc.conf_int(alpha=0.05)
    if hasattr(raw_ci, "iloc"):
        ci_lower = raw_ci.iloc[:, 0].values.astype(float)
        ci_upper = raw_ci.iloc[:, 1].values.astype(float)
    else:
        ci_arr   = np.array(raw_ci, dtype=float)
        ci_lower = ci_arr[:, 0]
        ci_upper = ci_arr[:, 1]

    last_date    = pd.Timestamp(series["ds"].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")

    fc_df = pd.DataFrame({
        "ds":         future_dates,
        "yhat":       pred_mean,
        "yhat_lower": ci_lower,
        "yhat_upper": ci_upper,
    })

    return pd.concat([hist_df, fc_df], ignore_index=True)


def monthly_forecast(daily_forecast: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily forecast to monthly for display."""
    df = daily_forecast.copy()
    df["ds"]    = pd.to_datetime(df["ds"])
    df["month"] = df["ds"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        yhat=("yhat",       "sum"),
        yhat_lower=("yhat_lower", "sum"),
        yhat_upper=("yhat_upper", "sum"),
    ).reset_index()
    monthly["month_str"] = monthly["month"].dt.strftime("%b %Y")
    return monthly


def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute MAPE and R2 between actual and fitted values."""
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask   = actual > 0
    mape   = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"mape": round(float(mape), 2), "r2": round(float(r2), 3)}
"""
src/pipeline/data_generator.py
Generates a realistic synthetic e-commerce transactional dataset.
Run once: python -m src.pipeline.data_generator
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(seed=42)

CATEGORIES = ["Electronics", "Apparel", "Home", "Beauty", "Sports"]
CATEGORY_WEIGHTS = [0.35, 0.22, 0.18, 0.14, 0.11]
CATEGORY_AVG_PRICE = {"Electronics": 180, "Apparel": 65, "Home": 90, "Beauty": 45, "Sports": 75}

def _simulate_orders(n_orders: int = 47231) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01")
    end   = pd.Timestamp("2024-12-31")
    total_days = (end - start).days + 1

    # Seasonal demand curve: higher in Nov-Dec, dip in Feb
    day_index = np.arange(total_days)
    seasonal  = 1 + 0.3 * np.sin((day_index / total_days) * 2 * np.pi - np.pi / 2)
    seasonal += 0.4 * (day_index > 300)          # Black Friday / holiday spike
    seasonal  = np.maximum(seasonal, 0.5)
    day_probs = seasonal / seasonal.sum()

    order_days = RNG.choice(total_days, size=n_orders, p=day_probs)
    order_dates = pd.to_datetime([start + pd.Timedelta(days=int(d)) for d in order_days])

    categories = RNG.choice(CATEGORIES, size=n_orders, p=CATEGORY_WEIGHTS)
    base_prices = np.array([CATEGORY_AVG_PRICE[c] for c in categories], dtype=float)
    noise       = RNG.normal(1.0, 0.25, size=n_orders)
    prices      = np.round(np.maximum(base_prices * noise, 5.0), 2)

    quantities  = RNG.choice([1, 2, 3, 4, 5], size=n_orders, p=[0.55, 0.25, 0.10, 0.06, 0.04])
    order_value = np.round(prices * quantities, 2)

    # Inject anomalies (0.5% of orders — fraudulent/bot activity)
    n_anomalies = int(n_orders * 0.005)
    anomaly_idx = RNG.choice(n_orders, size=n_anomalies, replace=False)
    order_value[anomaly_idx] *= RNG.uniform(8, 20, size=n_anomalies)

    # Assign user_ids (12,840 unique users with realistic repeat-purchase distribution)
    n_users = 12840
    user_weights = RNG.pareto(1.5, size=n_users) + 1
    user_weights /= user_weights.sum()
    user_ids = RNG.choice(np.arange(1, n_users + 1), size=n_orders, p=user_weights)

    df = pd.DataFrame({
        "order_id":    np.arange(1, n_orders + 1),
        "order_date":  order_dates,
        "user_id":     user_ids,
        "category":    categories,
        "unit_price":  prices,
        "quantity":    quantities,
        "order_value": order_value,
        "is_anomaly":  np.isin(np.arange(n_orders), anomaly_idx).astype(int),
    })

    return df.sort_values("order_date").reset_index(drop=True)


def generate_and_save() -> Path:
    out = Path("data/orders.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = _simulate_orders()
    df.to_csv(out, index=False)
    print(f"[DataGenerator] Saved {len(df):,} orders → {out}")
    return out


if __name__ == "__main__":
    generate_and_save()

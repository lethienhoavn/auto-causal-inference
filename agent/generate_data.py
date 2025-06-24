import pandas as pd
import numpy as np
import sqlite3
import os

def generate_bank_dataset(db_path="data/bank.db"):
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n_samples),
        "income": np.random.normal(50000, 15000, size=n_samples).astype(int),
        "education": np.random.choice(["high_school", "bachelor", "master", "phd"], size=n_samples),
        "branch_visits": np.random.poisson(2, size=n_samples),
        "channel_preference": np.random.choice(["online", "phone", "in-branch"], size=n_samples),
        "customer_engagement": np.random.uniform(0, 1, size=n_samples),
        "region_code": np.random.choice(["north", "south", "east", "west"], size=n_samples),
        "promotion_offer": np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
        "activated_ib": 0  # placeholder
    })

    # Simulate treatment effect
    df["activated_ib"] = (
        0.1 * df["promotion_offer"] +
        0.02 * df["branch_visits"] +
        0.3 * df["customer_engagement"] +
        np.random.normal(0, 0.1, size=n_samples)
    ) > 0.4

    df["activated_ib"] = df["activated_ib"].astype(int)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("customer_data", conn, if_exists="replace", index=False)

    print(f"✅ Sample dataset saved to {db_path}")

if __name__ == "__main__":
    generate_bank_dataset()

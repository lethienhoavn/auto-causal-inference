import pandas as pd
import numpy as np
import sqlite3
import os


def generate_bank_dataset(db_path="data/bank.db"):
    np.random.seed(42)
    n_samples = 1000

    # Base variables
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n_samples),
        "income": np.random.normal(50000, 15000, size=n_samples).astype(int),
        "education": np.random.choice(["high_school", "bachelor", "master", "phd"], size=n_samples),
        "branch_visits": np.random.poisson(2, size=n_samples),
        "channel_preference": np.random.choice(["online", "phone", "in-branch"], size=n_samples),
        "region_code": np.random.choice(["north", "south", "east", "west"], size=n_samples),
        "promotion_offer": np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
    })

    # Add indirect relations
    # Income impacts engagement
    df["customer_engagement"] = (
        np.clip((df["income"] - 50000) / 40000, -0.5, 0.5) +
        np.random.normal(0, 0.1, size=n_samples)
    )

    # Education impacts branch visits
    edu_map = {"high_school": 2, "bachelor": 1, "master": -1, "phd": -2}
    df["branch_visits"] = np.maximum(
        df["branch_visits"] + df["education"].map(edu_map), 0
    )

    # Final outcome: activated_ib
    logits = (
        0.2 * df["promotion_offer"] +
        0.05 * df["branch_visits"] +
        0.5 * df["customer_engagement"] +
        np.random.normal(0, 0.2, size=n_samples)
    )

    df["activated_ib"] = (logits > 0.4).astype(int)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("customer_data", conn, if_exists="replace", index=False)

    print(f"✅ Bank dataset saved to {db_path}")
    print(df.head())


if __name__ == "__main__":
    generate_bank_dataset()

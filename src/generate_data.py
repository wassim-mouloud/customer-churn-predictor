"""Generate synthetic customer churn dataset with realistic patterns."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_churn_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic telecom customer churn dataset.

    Churn factors (realistic correlations):
    - Month-to-month contracts have higher churn (~40%)
    - Fiber optic without security has higher churn
    - High monthly charges with short tenure = high churn
    - Senior citizens have slightly higher churn
    - Customers without tech support churn more

    Args:
        n_samples: Number of customer records to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with customer features and churn labels
    """
    np.random.seed(random_state)

    gender = np.random.choice([0, 1], n_samples)  # 0=Female, 1=Male
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice([0, 1], n_samples, p=[0.52, 0.48])
    dependents = np.random.choice([0, 1], n_samples, p=[0.70, 0.30])


    tenure = np.clip(
        np.random.exponential(scale=24, size=n_samples),
        1, 72
    ).astype(int)

    contract_type = np.random.choice(
        [0, 1, 2], n_samples,
        p=[0.55, 0.25, 0.20]  
    )

    payment_method = np.random.choice(
        [0, 1, 2, 3], n_samples,
        p=[0.35, 0.20, 0.25, 0.20]  
    )

    paperless_billing = np.random.choice([0, 1], n_samples, p=[0.40, 0.60])

    internet_service = np.random.choice(
        [0, 1, 2], n_samples,
        p=[0.20, 0.35, 0.45]  
    )

    online_security = np.where(
        internet_service == 0, 0,
        np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
    )

    tech_support = np.where(
        internet_service == 0, 0,
        np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
    )

    streaming_tv = np.where(
        internet_service == 0, 0,
        np.random.choice([0, 1], n_samples, p=[0.50, 0.50])
    )

    streaming_movies = np.where(
        internet_service == 0, 0,
        np.random.choice([0, 1], n_samples, p=[0.50, 0.50])
    )

    base_charge = 20
    monthly_charges = (
        base_charge
        + internet_service * 25  
        + online_security * 10
        + tech_support * 10
        + streaming_tv * 12
        + streaming_movies * 12
        + np.random.normal(0, 5, n_samples)  
    )
    monthly_charges = np.clip(monthly_charges, 20, 120).round(2)

    total_charges = (tenure * monthly_charges * np.random.uniform(0.95, 1.05, n_samples)).round(2)

    churn_prob = np.full(n_samples, 0.15)  # Base probability

    churn_prob = np.where(contract_type == 0, churn_prob + 0.25, churn_prob)  # Month-to-month
    churn_prob = np.where(contract_type == 2, churn_prob - 0.12, churn_prob)  # Two year

    churn_prob = np.where(tenure < 12, churn_prob + 0.10, churn_prob)  # New customers
    churn_prob = np.where(tenure > 48, churn_prob - 0.15, churn_prob)  # Loyal customers

    churn_prob = np.where(payment_method == 0, churn_prob + 0.10, churn_prob)

    fiber_no_security = (internet_service == 2) & (online_security == 0)
    churn_prob = np.where(fiber_no_security, churn_prob + 0.12, churn_prob)

    no_support = (internet_service > 0) & (tech_support == 0)
    churn_prob = np.where(no_support, churn_prob + 0.08, churn_prob)

    high_charge_new = (monthly_charges > 70) & (tenure < 12)
    churn_prob = np.where(high_charge_new, churn_prob + 0.15, churn_prob)

    churn_prob = np.where(senior_citizen == 1, churn_prob + 0.05, churn_prob)

    churn_prob = np.where(partner == 1, churn_prob - 0.05, churn_prob)
    churn_prob = np.where(dependents == 1, churn_prob - 0.05, churn_prob)

    churn_prob = np.clip(churn_prob, 0.05, 0.85)

    churn = (np.random.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'tenure': tenure,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'paperless_billing': paperless_billing,
        'internet_service': internet_service,
        'online_security': online_security,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'churn': churn
    })

    from .config import (
        GENDER_LABELS, CONTRACT_LABELS, PAYMENT_LABELS, INTERNET_LABELS
    )

    df['gender_label'] = df['gender'].map(GENDER_LABELS)
    df['contract_label'] = df['contract_type'].map(CONTRACT_LABELS)
    df['payment_label'] = df['payment_method'].map(PAYMENT_LABELS)
    df['internet_label'] = df['internet_service'].map(INTERNET_LABELS)
    df['churn_label'] = df['churn'].map({0: 'No', 1: 'Yes'})

    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save dataset to CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")


if __name__ == "__main__":
    df = generate_churn_data()

    print(f"Generated {len(df)} customer records")
    print(f"Churn rate: {df['churn'].mean():.1%}")
    print(f"\nFeature summary:")
    print(df.describe())

    data_path = Path(__file__).parent.parent / "data" / "churn_data.csv"
    save_data(df, str(data_path))

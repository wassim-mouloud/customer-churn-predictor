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

    # Demographics
    gender = np.random.choice([0, 1], n_samples)  # 0=Female, 1=Male
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice([0, 1], n_samples, p=[0.52, 0.48])
    dependents = np.random.choice([0, 1], n_samples, p=[0.70, 0.30])

    # Account details
    # Tenure: exponential-like distribution (many new, fewer long-term)
    tenure = np.clip(
        np.random.exponential(scale=24, size=n_samples),
        1, 72
    ).astype(int)

    # Contract type: weighted distribution
    contract_type = np.random.choice(
        [0, 1, 2], n_samples,
        p=[0.55, 0.25, 0.20]  # Month-to-month, One year, Two year
    )

    # Payment method
    payment_method = np.random.choice(
        [0, 1, 2, 3], n_samples,
        p=[0.35, 0.20, 0.25, 0.20]  # Electronic check, Mailed, Bank, Credit
    )

    paperless_billing = np.random.choice([0, 1], n_samples, p=[0.40, 0.60])

    # Services
    internet_service = np.random.choice(
        [0, 1, 2], n_samples,
        p=[0.20, 0.35, 0.45]  # No, DSL, Fiber optic
    )

    # Service add-ons (only available if internet service exists)
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

    # Monthly charges based on services
    base_charge = 20
    monthly_charges = (
        base_charge
        + internet_service * 25  # DSL=$25, Fiber=$50
        + online_security * 10
        + tech_support * 10
        + streaming_tv * 12
        + streaming_movies * 12
        + np.random.normal(0, 5, n_samples)  # Random variation
    )
    monthly_charges = np.clip(monthly_charges, 20, 120).round(2)

    # Total charges = tenure * monthly (with some variation)
    total_charges = (tenure * monthly_charges * np.random.uniform(0.95, 1.05, n_samples)).round(2)

    # Calculate churn probability based on realistic factors
    churn_prob = np.full(n_samples, 0.15)  # Base probability

    # Contract type effect (biggest factor)
    churn_prob = np.where(contract_type == 0, churn_prob + 0.25, churn_prob)  # Month-to-month
    churn_prob = np.where(contract_type == 2, churn_prob - 0.12, churn_prob)  # Two year

    # Tenure effect
    churn_prob = np.where(tenure < 12, churn_prob + 0.10, churn_prob)  # New customers
    churn_prob = np.where(tenure > 48, churn_prob - 0.15, churn_prob)  # Loyal customers

    # Payment method (electronic check = higher churn)
    churn_prob = np.where(payment_method == 0, churn_prob + 0.10, churn_prob)

    # Fiber optic without security = frustration
    fiber_no_security = (internet_service == 2) & (online_security == 0)
    churn_prob = np.where(fiber_no_security, churn_prob + 0.12, churn_prob)

    # No tech support with internet = higher churn
    no_support = (internet_service > 0) & (tech_support == 0)
    churn_prob = np.where(no_support, churn_prob + 0.08, churn_prob)

    # High charges + short tenure = price sensitive new customers
    high_charge_new = (monthly_charges > 70) & (tenure < 12)
    churn_prob = np.where(high_charge_new, churn_prob + 0.15, churn_prob)

    # Senior citizens slightly higher churn
    churn_prob = np.where(senior_citizen == 1, churn_prob + 0.05, churn_prob)

    # Partners and dependents = more stable
    churn_prob = np.where(partner == 1, churn_prob - 0.05, churn_prob)
    churn_prob = np.where(dependents == 1, churn_prob - 0.05, churn_prob)

    # Clip probabilities
    churn_prob = np.clip(churn_prob, 0.05, 0.85)

    # Generate churn based on probability
    churn = (np.random.random(n_samples) < churn_prob).astype(int)

    # Create DataFrame
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

    # Add readable labels for display
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
    # Generate and save dataset
    df = generate_churn_data()

    # Print summary
    print(f"Generated {len(df)} customer records")
    print(f"Churn rate: {df['churn'].mean():.1%}")
    print(f"\nFeature summary:")
    print(df.describe())

    # Save to data directory
    data_path = Path(__file__).parent.parent / "data" / "churn_data.csv"
    save_data(df, data_path)

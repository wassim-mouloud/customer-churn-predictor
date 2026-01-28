"""Configuration constants for the Customer Churn Predictor."""

FEATURE_COLUMNS = [
    'tenure', 'monthly_charges', 'total_charges',
    'contract_type', 'payment_method', 'internet_service',
    'online_security', 'tech_support', 'streaming_tv',
    'streaming_movies', 'gender', 'senior_citizen',
    'partner', 'dependents', 'paperless_billing'
]

CONTRACT_OPTIONS = {
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}

PAYMENT_OPTIONS = {
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3
}

INTERNET_OPTIONS = {
    'No': 0,
    'DSL': 1,
    'Fiber optic': 2
}

GENDER_OPTIONS = {
    'Female': 0,
    'Male': 1
}

CONTRACT_LABELS = {v: k for k, v in CONTRACT_OPTIONS.items()}
PAYMENT_LABELS = {v: k for k, v in PAYMENT_OPTIONS.items()}
INTERNET_LABELS = {v: k for k, v in INTERNET_OPTIONS.items()}
GENDER_LABELS = {v: k for k, v in GENDER_OPTIONS.items()}

COLORS = {
    'background': '#0E1117',
    'paper': '#1E1E1E',
    'primary': '#00D4FF',
    'secondary': '#BB86FC',
    'success': '#00E676',
    'warning': '#FFD93D',
    'danger': '#FF4444',
    'text': '#FAFAFA',
    'text_secondary': '#888888',
    'grid': '#2D2D2D'
}

CHART_COLORS = ['#00D4FF', '#BB86FC', '#00E676', '#FFD93D', '#FF79C6', '#FF9F43']

RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 1.0
}

RISK_COLORS = {
    'low': '#00E676',
    'medium': '#FFD93D',
    'high': '#FF4444'
}

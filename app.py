"""Customer Churn Predictor - Streamlit Web Application."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.model import ChurnPredictor
from src.generate_data import generate_churn_data
from src.config import (
    COLORS, CONTRACT_OPTIONS, PAYMENT_OPTIONS, INTERNET_OPTIONS,
    GENDER_OPTIONS, RISK_THRESHOLDS, RISK_COLORS
)
from src.visualizations import (
    plot_churn_distribution, plot_tenure_distribution, plot_charges_scatter,
    plot_churn_by_contract, plot_churn_by_service, plot_correlation_heatmap,
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_feature_importance, plot_churn_gauge, plot_monthly_charges_distribution
)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{
        background-color: {COLORS['background']};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['paper']};
    }}

    /* Headers */
    h1, h2, h3 {{
        color: {COLORS['primary']} !important;
    }}

    /* Metric cards */
    [data-testid="stMetricValue"] {{
        color: {COLORS['text']} !important;
        font-size: 2rem !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS['text_secondary']} !important;
    }}

    /* Custom cards */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['paper']} 0%, #2D2D2D 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid {COLORS['primary']};
        margin: 10px 0;
    }}

    .risk-badge {{
        padding: 10px 20px;
        border-radius: 20px;
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
        display: inline-block;
    }}

    .risk-low {{
        background: linear-gradient(135deg, {RISK_COLORS['low']} 0%, #00C853 100%);
        color: #000;
    }}

    .risk-medium {{
        background: linear-gradient(135deg, {RISK_COLORS['medium']} 0%, #FFC107 100%);
        color: #000;
    }}

    .risk-high {{
        background: linear-gradient(135deg, {RISK_COLORS['high']} 0%, #D32F2F 100%);
        color: #fff;
    }}

    .prediction-result {{
        background: linear-gradient(135deg, {COLORS['paper']} 0%, #2D2D2D 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }}

    .factor-card {{
        background: {COLORS['paper']};
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 3px solid;
    }}

    .factor-high {{ border-color: {RISK_COLORS['high']}; }}
    .factor-medium {{ border-color: {RISK_COLORS['medium']}; }}
    .factor-low {{ border-color: {RISK_COLORS['low']}; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {COLORS['paper']};
        border-radius: 8px;
        padding: 10px 20px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']};
        color: white !important;
    }}

    .stTabs [data-baseweb="tab"] p {{
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

# Paths
DATA_PATH = Path(__file__).parent / "data" / "churn_data.csv"
MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"


@st.cache_resource
def load_model():
    """Load or train the churn prediction model."""
    predictor = ChurnPredictor()

    if MODEL_PATH.exists():
        predictor.load(MODEL_PATH)
    else:
        df = load_data()
        predictor.fit(df)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        predictor.save(MODEL_PATH)

    return predictor


@st.cache_data
def load_data():
    """Load or generate the churn dataset."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    else:
        df = generate_churn_data()
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return df


def get_risk_level(probability: float) -> tuple:
    """Get risk level and color based on probability."""
    if probability < RISK_THRESHOLDS['low']:
        return 'LOW', 'low'
    elif probability < RISK_THRESHOLDS['medium']:
        return 'MEDIUM', 'medium'
    else:
        return 'HIGH', 'high'


def get_risk_factors(features: dict, probability: float) -> list:
    """Analyze features and return risk factors."""
    factors = []

    # Contract type
    if features['contract_type'] == 0:
        factors.append({
            'factor': 'Month-to-month contract',
            'severity': 'high',
            'recommendation': 'Offer discount for annual commitment'
        })

    # Tenure
    if features['tenure'] < 12:
        factors.append({
            'factor': 'New customer (< 12 months)',
            'severity': 'medium',
            'recommendation': 'Engage with onboarding support and loyalty program'
        })

    # Payment method
    if features['payment_method'] == 0:
        factors.append({
            'factor': 'Electronic check payment',
            'severity': 'medium',
            'recommendation': 'Encourage automatic payment methods'
        })

    # Fiber without security
    if features['internet_service'] == 2 and features['online_security'] == 0:
        factors.append({
            'factor': 'Fiber optic without online security',
            'severity': 'high',
            'recommendation': 'Offer security package bundle discount'
        })

    # No tech support
    if features['internet_service'] > 0 and features['tech_support'] == 0:
        factors.append({
            'factor': 'No tech support with internet service',
            'severity': 'medium',
            'recommendation': 'Promote tech support add-on benefits'
        })

    # High charges + new customer
    if features['monthly_charges'] > 70 and features['tenure'] < 12:
        factors.append({
            'factor': 'High charges for new customer',
            'severity': 'high',
            'recommendation': 'Review pricing or offer promotional discount'
        })

    # Senior citizen
    if features['senior_citizen'] == 1:
        factors.append({
            'factor': 'Senior citizen customer',
            'severity': 'low',
            'recommendation': 'Offer senior-specific support and simplified plans'
        })

    # No partner or dependents
    if features['partner'] == 0 and features['dependents'] == 0:
        factors.append({
            'factor': 'Single household',
            'severity': 'low',
            'recommendation': 'Target with individual-focused promotions'
        })

    # Sort by severity
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    factors.sort(key=lambda x: severity_order[x['severity']])

    return factors[:5]  # Return top 5 factors


def predict_page(predictor: ChurnPredictor):
    """Render the prediction page."""
    st.title("🎯 Predict Customer Churn")
    st.markdown("Enter customer details to predict their likelihood of churning.")

    # Input form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Customer Info")
        gender = st.selectbox("Gender", options=list(GENDER_OPTIONS.keys()))
        senior_citizen = st.checkbox("Senior Citizen (65+)")
        partner = st.checkbox("Has Partner")
        dependents = st.checkbox("Has Dependents")

    with col2:
        st.markdown("#### 📋 Account Details")
        tenure = st.slider("Months as Customer", 1, 72, 12)
        contract = st.selectbox("Contract Type", options=list(CONTRACT_OPTIONS.keys()))
        payment = st.selectbox("Payment Method", options=list(PAYMENT_OPTIONS.keys()))
        paperless = st.checkbox("Paperless Billing", value=True)

    with col3:
        st.markdown("#### 🌐 Services")
        internet = st.selectbox("Internet Service", options=list(INTERNET_OPTIONS.keys()))

        # Conditional services
        if internet != 'No':
            online_security = st.checkbox("Online Security")
            tech_support = st.checkbox("Tech Support")
            streaming_tv = st.checkbox("Streaming TV")
            streaming_movies = st.checkbox("Streaming Movies")
        else:
            online_security = False
            tech_support = False
            streaming_tv = False
            streaming_movies = False

    # Charges section
    st.markdown("---")
    charge_col1, charge_col2 = st.columns(2)

    with charge_col1:
        monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 50.0, 0.5)

    with charge_col2:
        auto_calc = st.checkbox("Auto-calculate total charges", value=True)
        if auto_calc:
            total_charges = round(tenure * monthly_charges * 0.98, 2)
            st.metric("Total Charges", f"${total_charges:,.2f}")
        else:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges)

    # Build feature dictionary
    features = {
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': CONTRACT_OPTIONS[contract],
        'payment_method': PAYMENT_OPTIONS[payment],
        'internet_service': INTERNET_OPTIONS[internet],
        'online_security': int(online_security),
        'tech_support': int(tech_support),
        'streaming_tv': int(streaming_tv),
        'streaming_movies': int(streaming_movies),
        'gender': GENDER_OPTIONS[gender],
        'senior_citizen': int(senior_citizen),
        'partner': int(partner),
        'dependents': int(dependents),
        'paperless_billing': int(paperless)
    }

    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):

        # Make prediction
        prediction, probability = predictor.predict(**features)
        risk_level, risk_class = get_risk_level(probability)

        # Results section
        st.markdown("## Prediction Results")

        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            # Gauge chart
            gauge_fig = plot_churn_gauge(probability)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with result_col2:
            # Risk badge and summary
            st.markdown(f"""
            <div class="prediction-result">
                <div class="risk-badge risk-{risk_class}">{risk_level} RISK</div>
                <h2 style="color: {COLORS['text']}; margin-top: 20px;">
                    {'Will Likely Churn' if prediction else 'Likely to Stay'}
                </h2>
                <p style="color: {COLORS['text_secondary']};">
                    {probability:.1%} probability of churning
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Risk factors
        st.markdown("### 🔍 Key Risk Factors")
        risk_factors = get_risk_factors(features, probability)

        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"""
                <div class="factor-card factor-{factor['severity']}">
                    <strong>{factor['factor']}</strong>
                    <br><small style="color: {COLORS['text_secondary']};">
                    💡 {factor['recommendation']}
                    </small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No significant risk factors identified. Customer profile looks stable!")


def exploration_page(df: pd.DataFrame):
    """Render the data exploration page."""
    st.title("📊 Data Exploration")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        churn_rate = df['churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col3:
        st.metric("Avg Tenure", f"{df['tenure'].mean():.1f} mo")
    with col4:
        st.metric("Avg Monthly Charges", f"${df['monthly_charges'].mean():.2f}")

    st.markdown("---")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Churn Distribution",
        "Tenure Analysis",
        "Charges Analysis",
        "Contract Analysis",
        "Service Analysis",
        "Correlations"
    ])

    with tab1:
        fig = plot_churn_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_tenure_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_charges_scatter(df)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plot_monthly_charges_distribution(df)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = plot_churn_by_contract(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        fig = plot_churn_by_service(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        fig = plot_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)

    # Data preview
    with st.expander("📋 View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)


def performance_page(predictor: ChurnPredictor):
    """Render the model performance page."""
    st.title("📈 Model Performance")

    metrics = predictor.get_metrics()

    if not metrics:
        st.warning("Model metrics not available. Please retrain the model.")
        return

    # Metrics overview
    st.markdown("### Classification Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Training Set")
        train_col1, train_col2 = st.columns(2)
        with train_col1:
            st.metric("Accuracy", f"{metrics['train']['accuracy']:.1%}")
            st.metric("Precision", f"{metrics['train']['precision']:.1%}")
        with train_col2:
            st.metric("Recall", f"{metrics['train']['recall']:.1%}")
            st.metric("F1 Score", f"{metrics['train']['f1']:.1%}")

    with col2:
        st.markdown("#### Test Set")
        test_col1, test_col2 = st.columns(2)
        with test_col1:
            st.metric("Accuracy", f"{metrics['test']['accuracy']:.1%}")
            st.metric("Precision", f"{metrics['test']['precision']:.1%}")
        with test_col2:
            st.metric("Recall", f"{metrics['test']['recall']:.1%}")
            st.metric("F1 Score", f"{metrics['test']['f1']:.1%}")

        st.metric("ROC-AUC", f"{metrics['test']['roc_auc']:.3f}")

    st.markdown("---")

    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Confusion Matrix",
        "ROC Curve",
        "Precision-Recall",
        "Feature Importance"
    ])

    with tab1:
        fig = plot_confusion_matrix(metrics['test']['confusion_matrix'])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_roc_curve(
            metrics['roc_curve']['fpr'],
            metrics['roc_curve']['tpr'],
            metrics['test']['roc_auc']
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = plot_precision_recall_curve(
            metrics['pr_curve']['precision'],
            metrics['pr_curve']['recall']
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = plot_feature_importance(metrics['feature_importance'])
        st.plotly_chart(fig, use_container_width=True)

    # Model details
    with st.expander("🔧 Model Details"):
        st.markdown(f"""
        **Algorithm:** Random Forest Classifier

        **Hyperparameters:**
        - Number of trees: 100
        - Max depth: 10
        - Min samples split: 5
        - Min samples leaf: 2
        - Class weight: balanced

        **Training Data:**
        - Training samples: {metrics['n_train']:,}
        - Test samples: {metrics['n_test']:,}
        - Test split: 20%
        """)


def about_page():
    """Render the about page."""
    st.title("ℹ️ About")

    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {COLORS['primary']};">Customer Churn Predictor</h3>
        <p>
        A machine learning application that predicts customer churn using
        <strong>Random Forest Classification</strong> with an interactive
        dark-themed Streamlit interface and Plotly visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚀 Features")
        st.markdown("""
        - **Churn Prediction** - Predict individual customer churn risk
        - **Risk Assessment** - Color-coded risk levels with probability gauge
        - **Risk Factors** - Actionable insights for retention
        - **Data Exploration** - Interactive visualizations of customer data
        - **Model Performance** - Comprehensive classification metrics
        """)

    with col2:
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        - **Python 3.8+**
        - **scikit-learn** - Random Forest Classifier
        - **Streamlit** - Web interface
        - **Plotly** - Interactive visualizations
        - **Pandas & NumPy** - Data processing
        """)

    st.markdown("---")

    st.markdown("### 📊 Dataset")
    st.markdown("""
    The model is trained on a synthetic telecom customer dataset with 5,000 records.
    Features include customer demographics, account information, service subscriptions,
    and billing details. The target variable is customer churn (binary: Yes/No).
    """)

    st.markdown("### 🎯 Model Performance")
    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | Accuracy | ~80% |
    | Precision | ~65% |
    | Recall | ~75% |
    | F1 Score | ~70% |
    | ROC-AUC | ~85% |
    """)


def main():
    """Main application entry point."""
    # Load model and data
    predictor = load_model()
    df = load_data()

    # Sidebar navigation
    st.sidebar.title("🔮 Churn Predictor")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🎯 Predict Churn", "📊 Data Exploration", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="color: {COLORS['text_secondary']}; font-size: 0.8em;">
    Built with Streamlit & Plotly<br>
    Model: Random Forest
    </div>
    """, unsafe_allow_html=True)

    # Render selected page
    if page == "🎯 Predict Churn":
        predict_page(predictor)
    elif page == "📊 Data Exploration":
        exploration_page(df)
    elif page == "📈 Model Performance":
        performance_page(predictor)
    else:
        about_page()


if __name__ == "__main__":
    main()

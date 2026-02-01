# Customer Churn Predictor

A machine learning application that predicts customer churn using **Random Forest Classification** with an interactive dark-themed Streamlit interface and Plotly visualizations.

## Features

- Random Forest Classifier with scikit-learn
- Interactive dark-themed Streamlit interface
- Dynamic Plotly visualizations
- Churn probability gauge and risk assessment
- Actionable retention recommendations
- Comprehensive model evaluation metrics

## Demo

![Prediction Page](assets/demo-prediction.png)
![Data Exploration](assets/demo-exploration.png)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/churn-predictor.git
cd churn-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app (auto-generates data and trains model on first run)
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Project Structure

```
churn-predictor/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md
├── data/
│   └── churn_data.csv     # Generated dataset
├── models/
│   └── churn_model.joblib # Trained model
└── src/
    ├── config.py          # Constants and theme colors
    ├── generate_data.py   # Synthetic data generation
    ├── model.py           # ChurnPredictor class
    └── visualizations.py  # Plotly chart functions
```

## Dataset Features

| Category     | Features                                                                        |
| ------------ | ------------------------------------------------------------------------------- |
| Demographics | gender, senior_citizen, partner, dependents                                     |
| Account      | tenure, contract_type, payment_method, paperless_billing                        |
| Services     | internet_service, online_security, tech_support, streaming_tv, streaming_movies |
| Charges      | monthly_charges, total_charges                                                  |
| Target       | churn (~27% rate)                                                               |

## Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~80%  |
| Precision | ~65%  |
| Recall    | ~75%  |
| F1 Score  | ~70%  |
| ROC-AUC   | ~85%  |

## Technologies

- **Python 3.8+**
- **scikit-learn** - Random Forest Classifier
- **Streamlit** - Web interface
- **Plotly** - Interactive visualizations
- **Pandas & NumPy** - Data processing
- **Joblib** - Model persistence

## Key Learnings

- Classification problem solving with Random Forest
- Feature importance analysis and interpretation
- Building interactive ML dashboards

"""Random Forest Classifier for customer churn prediction."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from .config import FEATURE_COLUMNS


class ChurnPredictor:
    """Random Forest Classifier for predicting customer churn."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the churn predictor.

        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = FEATURE_COLUMNS
        self.is_fitted = False
        self._metrics = None

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and prepare features from dataframe."""
        return df[self.feature_columns].values

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """
        Train the model and return evaluation metrics.

        Args:
            df: DataFrame with features and 'churn' target column
            test_size: Proportion of data to use for testing

        Returns:
            Dictionary containing train/test metrics and curves
        """
        X = self.prepare_features(df)
        y = df['churn'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y  # type: ignore[arg-type]
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        def calc_metrics(y_true, y_pred):
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            }

        fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_proba)

        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_test_proba)

        feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))

        self._metrics = {
            'train': calc_metrics(y_train, y_train_pred),
            'test': {
                **calc_metrics(y_test, y_test_pred),
                'roc_auc': roc_auc_score(y_test, y_test_proba),
                'confusion_matrix': confusion_matrix(y_test, y_test_pred),
                'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
            },
            'feature_importance': feature_importance,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr_curve': {'precision': pr_precision, 'recall': pr_recall, 'thresholds': pr_thresholds},
            'n_train': len(y_train),
            'n_test': len(y_test),
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba
        }

        return self._metrics

    def predict(self, **features) -> tuple:
        """
        Predict churn for a single customer.

        Args:
            **features: Customer features as keyword arguments

        Returns:
            Tuple of (prediction, probability)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0, 1]

        return int(prediction), float(probability)

    def predict_proba(self, **features) -> np.ndarray:
        """
        Get probability distribution for a single customer.

        Returns:
            Array of [P(no churn), P(churn)]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)[0]

    def predict_batch(self, df: pd.DataFrame) -> tuple:
        """
        Predict churn for multiple customers.

        Returns:
            Tuple of (predictions array, probabilities array)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities

    def get_feature_importance(self) -> dict:
        """Return feature importance scores sorted by importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_metrics(self) -> dict | None:
        """Return stored metrics from training."""
        return self._metrics

    def save(self, path: str) -> None:
        """Save the model, scaler, and metadata to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metrics': self._metrics
        }
        joblib.dump(model_data, path)

    def load(self, path: str) -> None:
        """Load the model, scaler, and metadata from disk."""
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self._metrics = model_data.get('metrics')
        self.is_fitted = True


if __name__ == "__main__":
    from .generate_data import generate_churn_data

    print("Generating data...")
    df = generate_churn_data()

    print("Training model...")
    predictor = ChurnPredictor()
    metrics = predictor.fit(df)

    print("\n=== Model Performance ===")
    print(f"Training Accuracy: {metrics['train']['accuracy']:.3f}")
    print(f"Test Accuracy: {metrics['test']['accuracy']:.3f}")
    print(f"Test Precision: {metrics['test']['precision']:.3f}")
    print(f"Test Recall: {metrics['test']['recall']:.3f}")
    print(f"Test F1 Score: {metrics['test']['f1']:.3f}")
    print(f"Test ROC-AUC: {metrics['test']['roc_auc']:.3f}")

    print("\n=== Feature Importance ===")
    for feature, importance in predictor.get_feature_importance().items():
        print(f"  {feature}: {importance:.4f}")

    print("\n=== Sample Prediction ===")
    prediction, probability = predictor.predict(
        tenure=5,
        monthly_charges=80,
        total_charges=400,
        contract_type=0,  
        payment_method=0, 
        internet_service=2, 
        online_security=0,
        tech_support=0,
        streaming_tv=1,
        streaming_movies=1,
        gender=1,
        senior_citizen=0,
        partner=0,
        dependents=0,
        paperless_billing=1
    )
    print(f"Prediction: {'Churn' if prediction else 'No Churn'}")
    print(f"Probability: {probability:.1%}")

    model_path = Path(__file__).parent.parent / "models" / "churn_model.joblib"
    predictor.save(str(model_path))
    print(f"\nModel saved to {model_path}")

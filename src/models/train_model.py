"""
Model training and evaluation functions.

This module contains functions for training machine learning models,
evaluating their performance, and saving/loading trained models.
"""

import pandas as pd
import numpy as np
import joblib
import json
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    calibration_curve,
)
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)


class OutagePredictionModel:
    """
    Main model class for outage prediction.
    """

    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize the model.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.is_fitted = False

        # Initialize model based on type
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                random_state=kwargs.get("random_state", 42),
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 6),
                random_state=kwargs.get("random_state", 42),
            )
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=kwargs.get("random_state", 42),
                max_iter=kwargs.get("max_iter", 1000),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        test_size: float = 0.2,
        temporal_split: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training and testing.

        Args:
            df: Input dataframe with features and target
            target_col: Name of target column
            test_size: Proportion of data for testing
            temporal_split: Whether to split by time (True) or randomly (False)

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Remove non-feature columns
        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                target_col,
                "datetime",
                "isoDate",
                "latitude",
                "longitude",
                "coordinates",
                "_id",
                "location",
            ]
        ]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle missing values
        X = X.fillna(X.mean())

        if temporal_split and "datetime" in df.columns:
            # Split by time
            df_sorted = df.sort_values("datetime")
            split_idx = int(len(df_sorted) * (1 - test_size))

            train_idx = df_sorted.index[:split_idx]
            test_idx = df_sorted.index[split_idx:]

            X_train = X.loc[train_idx]
            X_test = X.loc[test_idx]
            y_train = y.loc[train_idx]
            y_test = y.loc[test_idx]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        self.feature_columns = feature_cols

        logger.info(
            f"Training set: {len(X_train)} samples, {len(feature_cols)} features"
        )
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(
            f"Target distribution in training: {y_train.value_counts().to_dict()}"
        )

        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info(f"Training {self.model_type} model...")

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        logger.info("Model training completed")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features for prediction

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features for prediction

        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
        }

        # Feature importance (if available)
        if hasattr(self.model, "feature_importances_"):
            feature_importance = dict(
                zip(self.feature_columns, self.model.feature_importances_)
            )
            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            metrics["feature_importance"] = feature_importance

        logger.info(
            f"Model evaluation completed. ROC-AUC: {metrics['roc_auc_score']:.3f}"
        )

        return metrics

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Target
            cv: Number of folds

        Returns:
            Cross-validation scores
        """
        if not self.is_fitted:
            # Fit a temporary model for cross-validation
            temp_model = self.model
        else:
            temp_model = self.model

        scores = cross_val_score(temp_model, X, y, cv=cv, scoring="roc_auc")

        cv_results = {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores.tolist(),
        }

        logger.info(
            f"Cross-validation ROC-AUC: {cv_results['mean_score']:.3f} (+/- {cv_results['std_score']:.3f})"
        )

        return cv_results

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters using grid search.

        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for search

        Returns:
            Best parameters and scores
        """
        if param_grid is None:
            # Default parameter grids
            if self.model_type == "random_forest":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                }
            elif self.model_type == "gradient_boosting":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 10],
                    "learning_rate": [0.01, 0.1, 0.2],
                }
            elif self.model_type == "logistic_regression":
                param_grid = {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"],
                }

        logger.info(f"Tuning hyperparameters for {self.model_type}...")

        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True

        results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
        }

        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.3f}")

        return results

    def save_model(self, filepath: Path):
        """
        Save the trained model.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path):
        """
        Load a trained model.

        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.model_type = model_data["model_type"]
        self.feature_columns = model_data["feature_columns"]
        self.is_fitted = model_data["is_fitted"]

        logger.info(f"Model loaded from {filepath}")


def train_baseline_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> Dict[str, Dict]:
    """
    Train multiple baseline models and compare performance.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary of model results
    """
    models = {
        "random_forest": OutagePredictionModel("random_forest"),
        "gradient_boosting": OutagePredictionModel("gradient_boosting"),
        "logistic_regression": OutagePredictionModel("logistic_regression"),
    }

    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")

        # Train model
        model.train(X_train, y_train)

        # Evaluate model
        metrics = model.evaluate(X_test, y_test)

        results[name] = {"model": model, "metrics": metrics}

    # Find best model
    best_model_name = max(
        results.keys(), key=lambda k: results[k]["metrics"]["roc_auc_score"]
    )
    logger.info(
        f"Best model: {best_model_name} (ROC-AUC: {results[best_model_name]['metrics']['roc_auc_score']:.3f})"
    )

    return results

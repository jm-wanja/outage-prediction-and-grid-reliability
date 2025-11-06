"""
Model training utilities with hyperparameter optimization.

This module provides utilities for training machine learning models
with automated hyperparameter tuning using Optuna.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Install with: pip install optuna")

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train and optimize machine learning models for outage prediction.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}

    def get_model(
        self, model_type: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get a model instance with specified parameters.

        Args:
            model_type: Type of model to create
            params: Model parameters

        Returns:
            Model instance
        """
        if params is None:
            params = {}

        params["random_state"] = self.random_state

        if model_type == "random_forest":
            return RandomForestClassifier(**params)
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(**params)
        elif model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, **params)
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            params["random_state"] = params.pop("random_state", self.random_state)
            return xgb.XGBClassifier(**params)
        elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            params["random_state"] = params.pop("random_state", self.random_state)
            return lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Unknown or unavailable model type: {model_type}")

    def get_default_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a model type.

        Args:
            model_type: Type of model

        Returns:
            Dictionary of default parameters
        """
        params = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": "balanced",
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "subsample": 0.8,
                # Note: GradientBoostingClassifier doesn't support class_weight directly
                # But we can use sample_weight in fit() or undersample majority class
            },
            "logistic_regression": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "class_weight": "balanced",
            },
            "xgboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 1,
            },
            "lightgbm": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "class_weight": "balanced",
            },
        }
        return params.get(model_type, {})

    def get_param_space(self, model_type: str) -> Dict[str, Any]:
        """
        Get Optuna parameter space for hyperparameter optimization.

        Args:
            model_type: Type of model

        Returns:
            Dictionary defining parameter search space
        """
        spaces = {
            "random_forest": {
                "n_estimators": ("int", 50, 300),
                "max_depth": ("int", 3, 20),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 10),
                "max_features": ("categorical", ["sqrt", "log2", None]),
            },
            "gradient_boosting": {
                "n_estimators": ("int", 50, 300),
                "learning_rate": ("float", 0.01, 0.3),
                "max_depth": ("int", 3, 10),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 10),
                "subsample": ("float", 0.6, 1.0),
            },
            "logistic_regression": {
                "C": ("float", 0.001, 10.0),
                "penalty": ("categorical", ["l1", "l2"]),
                "solver": ("categorical", ["liblinear", "saga"]),
            },
            "xgboost": {
                "n_estimators": ("int", 50, 300),
                "learning_rate": ("float", 0.01, 0.3),
                "max_depth": ("int", 3, 10),
                "min_child_weight": ("int", 1, 10),
                "subsample": ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.6, 1.0),
                "gamma": ("float", 0, 5),
            },
            "lightgbm": {
                "n_estimators": ("int", 50, 300),
                "learning_rate": ("float", 0.01, 0.3),
                "max_depth": ("int", 3, 10),
                "num_leaves": ("int", 20, 100),
                "min_child_samples": ("int", 5, 50),
                "subsample": ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.6, 1.0),
            },
        }
        return spaces.get(model_type, {})

    def train_baseline(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cv_folds: int = 5,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a baseline model with default parameters.

        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            cv_folds: Number of cross-validation folds

        Returns:
            Trained model and validation metrics
        """
        logger.info(f"Training baseline {model_type} model...")

        # Get default parameters
        params = self.get_default_params(model_type)

        # For XGBoost, calculate and set scale_pos_weight for class imbalance
        if model_type == "xgboost":
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            if pos_count > 0:
                scale_pos_weight = neg_count / pos_count
                params["scale_pos_weight"] = scale_pos_weight
                logger.info(
                    f"XGBoost scale_pos_weight set to {scale_pos_weight:.2f} "
                    f"(neg={neg_count}, pos={pos_count})"
                )

        model = self.get_model(model_type, params)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        metrics = {}

        # Cross-validation score
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_state
        )
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        metrics["cv_roc_auc_mean"] = cv_scores.mean()
        metrics["cv_roc_auc_std"] = cv_scores.std()

        # Training metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = (
            model.predict_proba(X_train)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)
        metrics["train_precision"] = precision_score(
            y_train, y_train_pred, zero_division=0
        )
        metrics["train_recall"] = recall_score(y_train, y_train_pred, zero_division=0)
        metrics["train_f1"] = f1_score(y_train, y_train_pred, zero_division=0)

        if y_train_proba is not None:
            metrics["train_roc_auc"] = roc_auc_score(y_train, y_train_proba)
            metrics["train_avg_precision"] = average_precision_score(
                y_train, y_train_proba
            )

        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_proba = (
                model.predict_proba(X_val)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            metrics["val_accuracy"] = accuracy_score(y_val, y_val_pred)
            metrics["val_precision"] = precision_score(
                y_val, y_val_pred, zero_division=0
            )
            metrics["val_recall"] = recall_score(y_val, y_val_pred, zero_division=0)
            metrics["val_f1"] = f1_score(y_val, y_val_pred, zero_division=0)

            if y_val_proba is not None:
                metrics["val_roc_auc"] = roc_auc_score(y_val, y_val_proba)
                metrics["val_avg_precision"] = average_precision_score(
                    y_val, y_val_proba
                )

        self.models[model_type] = model
        self.cv_scores[model_type] = metrics

        logger.info(
            f"Baseline {model_type} - CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}"
        )
        if "val_roc_auc" in metrics:
            logger.info(
                f"Baseline {model_type} - Val ROC-AUC: {metrics['val_roc_auc']:.4f}"
            )

        return model, metrics

    def optimize_hyperparameters(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
        cv_folds: int = 5,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            model_type: Type of model to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds

        Returns:
            Best model, best parameters, and validation metrics
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Using baseline model.")
            return self.train_baseline(
                model_type, X_train, y_train, X_val, y_val, cv_folds
            )

        logger.info(
            f"Optimizing {model_type} hyperparameters with {n_trials} trials..."
        )

        param_space = self.get_param_space(model_type)

        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                param_type = param_config[0]

                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_config[1], param_config[2]
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, param_config[1], param_config[2], log=True
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config[1]
                    )

            # Special handling for logistic regression solver compatibility
            if model_type == "logistic_regression":
                if params.get("penalty") == "l1" and params.get("solver") not in [
                    "liblinear",
                    "saga",
                ]:
                    params["solver"] = "liblinear"

            # Create and train model
            model = self.get_model(model_type, params)

            # Use cross-validation for more robust evaluation
            cv = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=self.random_state
            )
            scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )

            return scores.mean()

        # Run optimization
        study = optuna.create_study(
            direction="maximize", study_name=f"{model_type}_optimization"
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Train final model with best parameters
        best_params = study.best_params
        logger.info(f"Best parameters for {model_type}: {best_params}")
        logger.info(f"Best CV ROC-AUC: {study.best_value:.4f}")

        best_model = self.get_model(model_type, best_params)
        best_model.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        y_val_proba = (
            best_model.predict_proba(X_val)[:, 1]
            if hasattr(best_model, "predict_proba")
            else None
        )

        metrics = {
            "cv_roc_auc": study.best_value,
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
            "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
            "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
        }

        if y_val_proba is not None:
            metrics["val_roc_auc"] = roc_auc_score(y_val, y_val_proba)
            metrics["val_avg_precision"] = average_precision_score(y_val, y_val_proba)

        self.models[f"{model_type}_optimized"] = best_model
        self.best_params[model_type] = best_params
        self.cv_scores[f"{model_type}_optimized"] = metrics

        logger.info(
            f"Optimized {model_type} - Val ROC-AUC: {metrics.get('val_roc_auc', 'N/A'):.4f}"
        )

        return best_model, best_params, metrics

    def save_model(
        self,
        model: Any,
        model_name: str,
        output_dir: Path,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        feature_names: Optional[list] = None,
    ):
        """
        Save trained model and metadata.

        Args:
            model: Trained model
            model_name: Name for the model
            output_dir: Directory to save model
            params: Model parameters
            metrics: Model metrics
            feature_names: List of feature names
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save metadata
        metadata = {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "parameters": params or {},
            "metrics": metrics or {},
            "feature_names": feature_names or [],
        }

        metadata_path = output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

    @staticmethod
    def load_model(model_path: Path) -> Any:
        """
        Load a trained model.

        Args:
            model_path: Path to model file

        Returns:
            Loaded model
        """
        return joblib.load(model_path)

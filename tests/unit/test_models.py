"""Unit tests for model training and evaluation functions."""

import pytest
import pandas as pd
import numpy as np
from src.models.train_model import OutagePredictionModel, train_baseline_models


class TestOutagePredictionModel:
    """Test the OutagePredictionModel class."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = OutagePredictionModel("random_forest")
        assert model.model_type == "random_forest"
        assert model.is_fitted is False

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError):
            OutagePredictionModel("invalid_model")

    def test_prepare_data(self, sample_processed_data):
        """Test data preparation."""
        model = OutagePredictionModel("random_forest")
        X_train, X_test, y_train, y_test = model.prepare_data(sample_processed_data)

        # Check shapes
        assert len(X_train) + len(X_test) == len(sample_processed_data)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        # Check that feature columns are set
        assert model.feature_columns is not None
        assert len(model.feature_columns) > 0

    def test_train_and_predict(self, sample_processed_data):
        """Test model training and prediction."""
        model = OutagePredictionModel(
            "random_forest", n_estimators=10
        )  # Small for speed
        X_train, X_test, y_train, y_test = model.prepare_data(sample_processed_data)

        # Train model
        model.train(X_train, y_train)
        assert model.is_fitted is True

        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})

        # Get probabilities
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_evaluate(self, sample_processed_data):
        """Test model evaluation."""
        model = OutagePredictionModel("random_forest", n_estimators=10)
        X_train, X_test, y_train, y_test = model.prepare_data(sample_processed_data)

        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        # Check that all expected metrics are present
        expected_keys = ["classification_report", "confusion_matrix", "roc_auc_score"]
        for key in expected_keys:
            assert key in metrics

    def test_save_and_load_model(self, sample_processed_data, temp_model_path):
        """Test model saving and loading."""
        model = OutagePredictionModel("random_forest", n_estimators=10)
        X_train, X_test, y_train, y_test = model.prepare_data(sample_processed_data)

        # Train and save
        model.train(X_train, y_train)
        model.save_model(temp_model_path)

        # Load new model
        new_model = OutagePredictionModel("random_forest")
        new_model.load_model(temp_model_path)

        # Check that model is properly loaded
        assert new_model.is_fitted is True
        assert new_model.model_type == "random_forest"
        assert new_model.feature_columns == model.feature_columns

        # Check that predictions are the same
        pred_original = model.predict(X_test)
        pred_loaded = new_model.predict(X_test)
        np.testing.assert_array_equal(pred_original, pred_loaded)


class TestTrainBaselineModels:
    """Test baseline model training function."""

    def test_train_baseline_models(self, sample_processed_data):
        """Test training multiple baseline models."""
        model = OutagePredictionModel("random_forest")
        X_train, X_test, y_train, y_test = model.prepare_data(sample_processed_data)

        results = train_baseline_models(X_train, X_test, y_train, y_test)

        # Check that all expected models are trained
        expected_models = ["random_forest", "gradient_boosting", "logistic_regression"]
        for model_name in expected_models:
            assert model_name in results
            assert "model" in results[model_name]
            assert "metrics" in results[model_name]

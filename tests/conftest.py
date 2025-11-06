"""Test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_outage_data():
    """Create sample outage data for testing."""
    np.random.seed(42)

    n_samples = 100
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")

    data = []
    for i in range(n_samples):
        data.append(
            {
                "datetime": np.random.choice(dates),
                "latitude": -1.2921 + np.random.normal(0, 0.5),
                "longitude": 36.8219 + np.random.normal(0, 0.5),
                "location": f"Location_{i}",
                "_id": f"id_{i}",
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def sample_processed_data(sample_outage_data):
    """Create sample processed data with features."""
    df = sample_outage_data.copy()

    # Add some features
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["cluster_id"] = np.random.randint(0, 5, len(df))
    df["target"] = np.random.binomial(1, 0.3, len(df))

    return df


@pytest.fixture
def temp_model_path(tmp_path):
    """Create temporary path for model saving/loading tests."""
    return tmp_path / "test_model.joblib"

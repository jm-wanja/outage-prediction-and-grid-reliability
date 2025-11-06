"""Unit tests for feature engineering functions."""

import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    create_temporal_features,
    perform_geographic_clustering,
    create_aggregation_features,
    create_target_variable,
)


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_create_temporal_features(self, sample_outage_data):
        """Test temporal feature creation."""
        df_with_features = create_temporal_features(sample_outage_data)

        # Check that all expected temporal features are created
        expected_features = [
            "year",
            "month",
            "day",
            "day_of_week",
            "day_of_year",
            "week_of_year",
            "quarter",
            "is_weekend",
            "season",
        ]

        for feature in expected_features:
            assert feature in df_with_features.columns

        # Check that is_weekend is binary
        assert set(df_with_features["is_weekend"].unique()).issubset({0, 1})

        # Check that season values are correct
        assert set(df_with_features["season"].unique()).issubset(
            {"dry", "long_rains", "short_rains"}
        )

    def test_perform_geographic_clustering(self, sample_outage_data):
        """Test geographic clustering."""
        df_clustered, metadata = perform_geographic_clustering(
            sample_outage_data, n_clusters=3, method="kmeans"
        )

        # Check that cluster_id column is added
        assert "cluster_id" in df_clustered.columns

        # Check metadata
        assert "method" in metadata
        assert "n_clusters" in metadata
        assert "cluster_centers" in metadata
        assert "silhouette_score" in metadata

        assert metadata["method"] == "kmeans"
        assert metadata["n_clusters"] <= 3  # May be less due to data size

    def test_create_aggregation_features(self, sample_processed_data):
        """Test aggregation feature creation."""
        # Need to sort by datetime first
        sample_processed_data = sample_processed_data.sort_values("datetime")

        df_with_agg = create_aggregation_features(sample_processed_data)

        # Check that aggregation features are created
        expected_features = [
            "outage_count_7d",
            "outage_rate_7d",
            "outages_last_week",
            "outages_last_month",
            "days_since_last_outage",
        ]

        for feature in expected_features:
            assert feature in df_with_agg.columns

    def test_create_target_variable(self, sample_processed_data):
        """Test target variable creation."""
        df_with_target = create_target_variable(sample_processed_data)

        # Check that target column is created
        assert "target" in df_with_target.columns

        # Check that target is binary
        assert set(df_with_target["target"].unique()).issubset({0, 1})

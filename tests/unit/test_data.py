"""Unit tests for data loading and processing functions."""

import pytest
import pandas as pd
import numpy as np
from src.data.load_data import load_raw_data, preprocess_data


class TestLoadData:
    """Test data loading functions."""

    def test_load_raw_data_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent_file.json")

    def test_preprocess_data_basic(self, sample_outage_data):
        """Test basic preprocessing functionality."""
        processed_df = preprocess_data(sample_outage_data)

        # Check that datetime column is properly converted
        assert pd.api.types.is_datetime64_any_dtype(processed_df["datetime"])

        # Check that latitude and longitude are numeric
        assert pd.api.types.is_numeric_dtype(processed_df["latitude"])
        assert pd.api.types.is_numeric_dtype(processed_df["longitude"])

        # Check that no rows are dropped for valid data
        assert len(processed_df) == len(sample_outage_data)

    def test_preprocess_data_with_missing_coords(self, sample_outage_data):
        """Test preprocessing with missing coordinates."""
        # Add some missing coordinates
        sample_data = sample_outage_data.copy()
        sample_data.loc[0, "latitude"] = np.nan
        sample_data.loc[1, "longitude"] = np.nan

        processed_df = preprocess_data(sample_data)

        # Should still process without error
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) <= len(
            sample_data
        )  # May drop rows with invalid coords

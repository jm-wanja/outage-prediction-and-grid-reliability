"""
Data preprocessing module for KPLC outage data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw outage data.

    Args:
        df: Raw DataFrame with interruption data

    Returns:
        Preprocessed DataFrame
    """
    logger.info("Starting data preprocessing...")

    # Make a copy to avoid modifying original
    df_processed = df.copy()

    # Extract coordinates from nested dictionary if they exist
    if "coordinates" in df_processed.columns:
        # Extract latitude and longitude from coordinates dictionary
        df_processed["latitude"] = df_processed["coordinates"].apply(
            lambda x: x.get("latitude") if isinstance(x, dict) else None
        )
        df_processed["longitude"] = df_processed["coordinates"].apply(
            lambda x: x.get("longitude") if isinstance(x, dict) else None
        )

    # Convert isoDate to datetime if it exists
    if "isoDate" in df_processed.columns:
        df_processed["datetime"] = pd.to_datetime(df_processed["isoDate"])
        df_processed["year"] = df_processed["datetime"].dt.year
        df_processed["month"] = df_processed["datetime"].dt.month
        df_processed["day_of_week"] = df_processed["datetime"].dt.dayofweek
        df_processed["quarter"] = df_processed["datetime"].dt.quarter

    # Clean coordinates if they exist
    if "latitude" in df_processed.columns and "longitude" in df_processed.columns:
        df_processed = clean_coordinates(df_processed)

    logger.info(f"Preprocessing complete. Shape: {df_processed.shape}")
    return df_processed


def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate geographic coordinates.

    Args:
        df: DataFrame with latitude and longitude columns

    Returns:
        DataFrame with cleaned coordinates
    """
    df_clean = df.copy()

    # Remove invalid coordinates
    if "latitude" in df_clean.columns:
        df_clean = df_clean[
            (df_clean["latitude"].notna()) & (df_clean["latitude"].between(-90, 90))
        ]

    if "longitude" in df_clean.columns:
        df_clean = df_clean[
            (df_clean["longitude"].notna()) & (df_clean["longitude"].between(-180, 180))
        ]

    # Kenya-specific bounds (approximately)
    # Latitude: -4.5 to 5.0, Longitude: 33.5 to 42.0
    if "latitude" in df_clean.columns and "longitude" in df_clean.columns:
        df_clean = df_clean[
            (df_clean["latitude"].between(-5, 6))
            & (df_clean["longitude"].between(33, 43))
        ]

    logger.info(f"Coordinate cleaning complete. Remaining records: {len(df_clean)}")
    return df_clean


def extract_temporal_features(
    df: pd.DataFrame, date_column: str = "datetime"
) -> pd.DataFrame:
    """
    Extract temporal features from datetime column.

    Args:
        df: DataFrame with datetime column
        date_column: Name of the datetime column

    Returns:
        DataFrame with additional temporal features
    """
    df_features = df.copy()

    if date_column not in df_features.columns:
        logger.warning(f"Column {date_column} not found")
        return df_features

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df_features[date_column]):
        df_features[date_column] = pd.to_datetime(df_features[date_column])

    # Extract features
    df_features["year"] = df_features[date_column].dt.year
    df_features["month"] = df_features[date_column].dt.month
    df_features["day"] = df_features[date_column].dt.day
    df_features["day_of_week"] = df_features[date_column].dt.dayofweek
    df_features["week_of_year"] = df_features[date_column].dt.isocalendar().week
    df_features["quarter"] = df_features[date_column].dt.quarter
    df_features["is_weekend"] = df_features["day_of_week"].isin([5, 6]).astype(int)

    # Season (for Kenya - two rainy seasons)
    # Long rains: March-May, Short rains: October-December
    df_features["season"] = df_features["month"].map(
        {
            3: "long_rains",
            4: "long_rains",
            5: "long_rains",
            10: "short_rains",
            11: "short_rains",
            12: "short_rains",
            1: "dry",
            2: "dry",
            6: "dry",
            7: "dry",
            8: "dry",
            9: "dry",
        }
    )

    logger.info("Temporal features extracted successfully")
    return df_features

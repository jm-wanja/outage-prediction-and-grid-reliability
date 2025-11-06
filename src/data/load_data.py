"""Data loading and validation utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load raw JSON data from file.

    Args:
        file_path: Path to JSON file

    Returns:
        List of dictionaries containing raw data

    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading raw data from {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} raw records")
    return data


def load_kplc_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load KPLC interruption data from JSON file.

    Args:
        file_path: Path to the JSON file containing KPLC data

    Returns:
        DataFrame with columns: latitude, longitude, datetime, timestamp

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the data format is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading KPLC data from {file_path}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    if not isinstance(data, list):
        raise ValueError("Expected JSON array format")

    # Extract and flatten the data
    records = []
    for i, record in enumerate(data):
        try:
            # Extract coordinates
            coords = record.get("coordinates", {})
            lat = coords.get("latitude")
            lon = coords.get("longitude")

            # Extract timestamps
            iso_date = record.get("isoDate")
            date_seconds = record.get("dateInSeconds")

            if lat is None or lon is None or iso_date is None:
                logger.warning(f"Skipping record {i} due to missing required fields")
                continue

            records.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "iso_date": iso_date,
                    "date_seconds": date_seconds,
                    "record_id": i,
                }
            )

        except Exception as e:
            logger.warning(f"Error processing record {i}: {e}")
            continue

    if not records:
        raise ValueError("No valid records found in the data file")

    df = pd.DataFrame(records)

    # Convert ISO date to datetime
    df["datetime"] = pd.to_datetime(df["iso_date"])

    logger.info(f"Loaded {len(df)} records from {len(data)} total records")

    return df


def validate_data(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate the loaded data for basic quality checks.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "has_required_columns": all(
            col in df.columns for col in ["latitude", "longitude", "datetime"]
        ),
        "no_missing_coordinates": df[["latitude", "longitude"]].notna().all().all(),
        "valid_coordinate_ranges": (
            df["latitude"].between(-90, 90).all()
            and df["longitude"].between(-180, 180).all()
        ),
        "no_missing_timestamps": df["datetime"].notna().all(),
        "chronological_order": df["datetime"].is_monotonic_increasing,
        "reasonable_sample_size": len(df) >= 100,
    }

    # Kenya-specific coordinate validation (rough bounds)
    kenya_bounds = {"lat_min": -5.0, "lat_max": 6.0, "lon_min": 33.0, "lon_max": 42.0}

    validation_results["within_kenya_bounds"] = (
        df["latitude"].between(kenya_bounds["lat_min"], kenya_bounds["lat_max"]).all()
        and df["longitude"]
        .between(kenya_bounds["lon_min"], kenya_bounds["lon_max"])
        .all()
    )

    # Log validation results
    for check, result in validation_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"Validation check '{check}': {status}")

    return validation_results


def get_data_summary(df: pd.DataFrame) -> Dict[str, Union[int, str, float]]:
    """Get a summary of the dataset.

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_records": len(df),
        "date_range_start": df["datetime"].min().strftime("%Y-%m-%d"),
        "date_range_end": df["datetime"].max().strftime("%Y-%m-%d"),
        "unique_coordinates": len(df[["latitude", "longitude"]].drop_duplicates()),
        "lat_min": df["latitude"].min(),
        "lat_max": df["latitude"].max(),
        "lon_min": df["longitude"].min(),
        "lon_max": df["longitude"].max(),
        "records_per_day": len(df) / (df["datetime"].max() - df["datetime"].min()).days,
    }

    return summary

"""
Data loader for KPLC interruption data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)


def load_interruption_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load KPLC interruption data from JSON file.

    Args:
        file_path: Path to the JSON data file

    Returns:
        DataFrame with interruption data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading interruption data from {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Convert isoDate to datetime
    if "isoDate" in df.columns:
        df["datetime"] = pd.to_datetime(df["isoDate"])
        df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} interruption records")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df

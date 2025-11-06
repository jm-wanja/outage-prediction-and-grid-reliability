"""Configuration settings for the outage prediction project."""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data file paths
KPLC_DATA_PATH = DATA_DIR / "kplc_interruption_data.json"

# Model configuration
DEFAULT_MODEL_CONFIG = {
    "clustering": {
        "algorithm": "kmeans",
        "n_clusters": 50,
        "random_state": 42,
        "max_iter": 300,
    },
    "features": {
        "temporal_features": ["hour", "day_of_week", "month", "season"],
        "spatial_features": ["cluster_id", "lat", "lon"],
        "lag_features": [1, 7, 30],  # days
        "rolling_windows": [7, 14, 30],  # days
    },
    "model": {
        "algorithm": "lightgbm",
        "hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "num_leaves": 31,
            "random_state": 42,
        },
        "cv_folds": 5,
        "test_size": 0.2,
        "validation_size": 0.2,
    },
    "prediction": {
        "horizon_days": 1,
        "threshold": 0.5,
    },
}

# Visualization configuration
VIZ_CONFIG = {
    "map": {
        "center_lat": -1.2921,  # Nairobi coordinates
        "center_lon": 36.8219,
        "default_zoom": 7,
    },
    "colors": {
        "low_risk": "#00ff00",
        "medium_risk": "#ffff00",
        "high_risk": "#ff0000",
    },
    "figure_size": (12, 8),
    "dpi": 300,
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

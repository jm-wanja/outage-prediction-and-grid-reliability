#!/usr/bin/env python3
"""
Main training script for the outage prediction model.

This script orchestrates the entire ML pipeline:
1. Load and preprocess data
2. Engineer features
3. Create target variable
4. Train models with hyperparameter optimization
5. Evaluate performance
6. Save results and visualizations

Usage:
    python scripts/train_model.py --data-path data/kplc_interruption_data.json
    python scripts/train_model.py --data-path data/kplc_interruption_data.json --optimize --n-trials 100
    python scripts/train_model.py --model-type xgboost --optimize
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import (
    create_temporal_features,
    perform_geographic_clustering,
)
from src.models.model_trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_target_variable(
    df: pd.DataFrame, window_days: int = 7, high_activity_threshold: int = 3
) -> pd.DataFrame:
    """
    Create binary target variable: will this cluster experience HIGH outage activity in the next N days?

    This reframes the problem to predict HIGH outage periods (more actionable than "any outage")
    which helps with class balance and provides more meaningful predictions for resource allocation.

    IMPORTANT: Counts unique OUTAGE DAYS, not individual outage records, to get meaningful thresholds.

    Args:
        df: DataFrame with outage data
        window_days: Number of days to look ahead (default: 7)
        high_activity_threshold: Number of UNIQUE OUTAGE DAYS to consider "high activity" (default: 3)

    Returns:
        DataFrame with target column (1 = high activity expected, 0 = normal activity)
    """
    logger.info(
        f"Creating target variable: predicting HIGH outage activity in next {window_days} days..."
    )
    logger.info(
        f"High activity defined as: >{high_activity_threshold} UNIQUE outage days in the window"
    )

    # Sort by datetime and cluster
    df = df.sort_values(["cluster_id", "datetime"]).copy()

    # Extract date (without time) for daily aggregation
    df["date_only"] = df["datetime"].dt.date

    # Add gap to prevent data leakage (don't predict same day)
    gap_days = 1

    # Initialize target
    df["target"] = 0
    df["future_outage_days"] = 0

    for cluster_id in df["cluster_id"].unique():
        cluster_mask = df["cluster_id"] == cluster_id
        cluster_indices = df[cluster_mask].index

        for idx in cluster_indices:
            current_date = df.loc[idx, "datetime"]

            # Define prediction window (start 1 day after to prevent leakage)
            window_start = current_date + pd.Timedelta(days=gap_days)
            window_end = current_date + pd.Timedelta(days=window_days)

            # Get future outages in this cluster within the window
            future_outages = df[
                (df["cluster_id"] == cluster_id)
                & (df["datetime"] > window_start)
                & (df["datetime"] <= window_end)
            ]

            # Count UNIQUE outage days (not individual records)
            unique_outage_days = future_outages["date_only"].nunique()
            df.loc[idx, "future_outage_days"] = unique_outage_days

            # Mark as high activity if exceeds threshold
            if unique_outage_days > high_activity_threshold:
                df.loc[idx, "target"] = 1

    target_counts = df["target"].value_counts()
    logger.info(f"Target distribution: {target_counts.to_dict()}")

    positive_ratio = target_counts.get(1, 0) / len(df)
    logger.info(f"Positive class ratio (high activity): {positive_ratio:.2%}")

    # Log statistics about future outage days
    logger.info("Future outage days statistics:")
    logger.info(f"  Mean: {df['future_outage_days'].mean():.2f}")
    logger.info(f"  Median: {df['future_outage_days'].median():.0f}")
    logger.info(f"  Max: {df['future_outage_days'].max():.0f}")
    logger.info(
        f"  Samples with >3 outage days: {(df['future_outage_days'] > 3).sum()}"
    )

    # Drop temporary date_only column before returning
    df = df.drop(columns=["date_only"])

    return df


def create_historical_features(
    df: pd.DataFrame, lookback_days: list = [7, 30]
) -> pd.DataFrame:
    """
    Create historical/lag features to provide temporal context for predictions.

    These features help the model understand recent outage patterns without data leakage.

    IMPORTANT: Counts unique OUTAGE DAYS, not individual records, for meaningful features.

    Args:
        df: DataFrame with outage data (must be sorted by datetime)
        lookback_days: List of lookback periods in days (default: [7, 30])

    Returns:
        DataFrame with added historical features
    """
    logger.info("Creating historical features for temporal context...")

    df = df.sort_values(["cluster_id", "datetime"]).copy()

    # Extract date (without time) for daily aggregation
    df["date_only"] = df["datetime"].dt.date

    # Initialize historical features
    for days in lookback_days:
        df[f"outages_last_{days}d"] = 0

    df["days_since_last_outage"] = 999  # High default value
    df["cluster_outage_rate_7d"] = 0.0
    df["cluster_outage_rate_30d"] = 0.0

    for cluster_id in df["cluster_id"].unique():
        cluster_mask = df["cluster_id"] == cluster_id
        cluster_indices = df[cluster_mask].index

        for idx in cluster_indices:
            current_date = df.loc[idx, "datetime"]

            # Count UNIQUE outage days in past lookback periods (for this cluster)
            for days in lookback_days:
                past_start = current_date - pd.Timedelta(days=days)
                past_outages = df[
                    (df["cluster_id"] == cluster_id)
                    & (
                        df["datetime"] < current_date
                    )  # Strict inequality to avoid leakage
                    & (df["datetime"] >= past_start)
                ]
                # Count unique days with outages, not total records
                unique_outage_days = past_outages["date_only"].nunique()
                df.loc[idx, f"outages_last_{days}d"] = unique_outage_days

                # Calculate cluster outage rate (unique outage days per calendar day)
                if days > 0:
                    df.loc[idx, f"cluster_outage_rate_{days}d"] = (
                        unique_outage_days / days
                    )

            # Days since last outage in this cluster
            previous_outages = df[
                (df["cluster_id"] == cluster_id) & (df["datetime"] < current_date)
            ]

            if len(previous_outages) > 0:
                last_outage_date = previous_outages["datetime"].max()
                days_since = (current_date - last_outage_date).days
                df.loc[idx, "days_since_last_outage"] = days_since

    # Log feature statistics
    logger.info("Historical feature statistics:")
    for days in lookback_days:
        col = f"outages_last_{days}d"
        logger.info(f"  {col}: mean={df[col].mean():.2f}, max={df[col].max():.0f}")

    logger.info(
        f"  days_since_last_outage: mean={df['days_since_last_outage'].mean():.1f}, "
        f"median={df['days_since_last_outage'].median():.0f}"
    )

    # Drop temporary date_only column before returning
    df = df.drop(columns=["date_only"])

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for model training.

    Args:
        df: DataFrame with features and target

    Returns:
        X (features), y (target), feature_names
    """
    # Define columns to exclude from features
    exclude_cols = [
        "target",
        "future_outage_days",  # This is derived from target - would cause leakage
        "datetime",
        "isoDate",
        "latitude",
        "longitude",
        "coordinates",
        "_id",
        "location",
        "month",  # month is categorical already handled
    ]

    # Select feature columns (including new historical features)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Prepare X and y
    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if categorical_cols:
        logger.info(f"Encoding categorical features: {categorical_cols}")
        # One-hot encode categorical variables
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        logger.info(f"After encoding: {X.shape[1]} features")

    # Handle any remaining missing values (only for numeric columns now)
    numeric_cols = X.select_dtypes(include=["number"]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # Get updated feature names after encoding
    feature_cols = X.columns.tolist()

    return X, y, feature_cols


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train outage prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline models on all types
  python scripts/train_model.py --data-path data/kplc_interruption_data.json

  # Train specific model with optimization
  python scripts/train_model.py --model-type xgboost --optimize --n-trials 100

  # Train all models with optimization
  python scripts/train_model.py --model-type all --optimize --n-trials 50
        """,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/kplc_interruption_data.json",
        help="Path to raw data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        choices=[
            "all",
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
            "xgboost",
            "lightgbm",
        ],
        help="Type of model to train",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Perform hyperparameter optimization using Optuna",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for optimization",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of geographic clusters (auto-detect if not specified)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proportion of data for test set"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Proportion of training data for validation set",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("OUTAGE PREDICTION MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    try:
        # ============================================================
        # Step 1: Load and preprocess data
        # ============================================================
        logger.info("\n[STEP 1/6] Loading and preprocessing data...")
        raw_data = load_raw_data(args.data_path)
        df = pd.DataFrame(raw_data)
        processed_data = preprocess_data(df)

        logger.info(f"Loaded {len(processed_data)} records")
        logger.info(
            f"Date range: {processed_data['datetime'].min()} to {processed_data['datetime'].max()}"
        )

        # ============================================================
        # Step 2: Feature engineering
        # ============================================================
        logger.info("\n[STEP 2/6] Engineering features...")

        # Temporal features
        data_with_temporal = create_temporal_features(processed_data)

        # Geographic clustering
        data_with_clusters, cluster_info = perform_geographic_clustering(
            data_with_temporal, n_clusters=args.n_clusters
        )

        logger.info(
            f"Created {data_with_clusters['cluster_id'].nunique()} geographic clusters"
        )

        # ============================================================
        # Step 3: Create target variable
        # ============================================================
        logger.info("\n[STEP 3/6] Creating target variable...")
        data_with_target = create_target_variable(
            data_with_clusters, window_days=7, high_activity_threshold=3
        )

        # Create historical features for temporal context
        logger.info("Adding historical features...")
        data_with_features = create_historical_features(
            data_with_target, lookback_days=[7, 30]
        )

        # ============================================================
        # Step 4: Prepare train/val/test splits
        # ============================================================
        logger.info("\n[STEP 4/6] Preparing data splits...")

        # Prepare features
        X, y, feature_names = prepare_features(data_with_features)

        # First split: train+val vs test (temporal split)
        data_sorted = data_with_features.sort_values("datetime")
        split_idx = int(len(data_sorted) * (1 - args.test_size))

        train_val_idx = data_sorted.index[:split_idx]
        test_idx = data_sorted.index[split_idx:]

        X_train_val = X.loc[train_val_idx]
        X_test = X.loc[test_idx]
        y_train_val = y.loc[train_val_idx]
        y_test = y.loc[test_idx]

        # Second split: train vs val (random split within train+val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=args.val_size,
            random_state=args.random_state,
            stratify=y_train_val,
        )

        logger.info(
            f"Training set: {len(X_train)} samples (positive: {y_train.sum() / len(y_train):.1%})"
        )
        logger.info(
            f"Validation set: {len(X_val)} samples (positive: {y_val.sum() / len(y_val):.1%})"
        )
        logger.info(
            f"Test set: {len(X_test)} samples (positive: {y_test.sum() / len(y_test):.1%})"
        )

        # ============================================================
        # Step 5: Train models
        # ============================================================
        logger.info("\n[STEP 5/6] Training models...")

        trainer = ModelTrainer(random_state=args.random_state)
        trained_models = {}
        all_metrics = {}

        # Determine which models to train
        if args.model_type == "all":
            model_types = [
                "random_forest",
                "gradient_boosting",
                "logistic_regression",
                "xgboost",
                "lightgbm",
            ]
        else:
            model_types = [args.model_type]

        # Train each model
        for model_type in model_types:
            logger.info(f"\n--- Training {model_type.upper()} ---")

            try:
                if args.optimize:
                    # Train with hyperparameter optimization
                    model, best_params, metrics = trainer.optimize_hyperparameters(
                        model_type=model_type,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        n_trials=args.n_trials,
                    )

                    # Save optimized model
                    trainer.save_model(
                        model=model,
                        model_name=f"{model_type}_optimized",
                        output_dir=output_dir,
                        params=best_params,
                        metrics=metrics,
                        feature_names=feature_names,
                    )

                    trained_models[f"{model_type}_optimized"] = model
                    all_metrics[f"{model_type}_optimized"] = metrics

                else:
                    # Train baseline model
                    model, metrics = trainer.train_baseline(
                        model_type=model_type,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                    )

                    # Save baseline model
                    params = trainer.get_default_params(model_type)
                    trainer.save_model(
                        model=model,
                        model_name=f"{model_type}_baseline",
                        output_dir=output_dir,
                        params=params,
                        metrics=metrics,
                        feature_names=feature_names,
                    )

                    trained_models[f"{model_type}_baseline"] = model
                    all_metrics[f"{model_type}_baseline"] = metrics

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {str(e)}")
                continue

        if not trained_models:
            raise ValueError("No models were successfully trained!")

        # ============================================================
        # Step 6: Evaluate and compare models
        # ============================================================
        logger.info("\n[STEP 6/6] Evaluating models on test set...")

        evaluator = ModelEvaluator()

        # Evaluate all models on test set
        test_metrics = {}
        for model_name, model in trained_models.items():
            test_metrics[model_name] = evaluator.evaluate_model(
                model=model, X_test=X_test, y_test=y_test, model_name=model_name
            )

        # Generate comprehensive evaluation report
        evaluator.generate_report(
            output_path=output_dir / "evaluation",
            models=trained_models,
            X_test=X_test,
            y_test=y_test,
        )

        # Save all metrics
        metrics_file = output_dir / "all_metrics.json"
        with open(metrics_file, "w") as f:
            # Combine training and test metrics
            combined_metrics = {}
            for model_name in trained_models.keys():
                combined_metrics[model_name] = {
                    "training": all_metrics[model_name],
                    "test": test_metrics[model_name],
                }
            json.dump(combined_metrics, f, indent=2, default=str)
        logger.info(f"All metrics saved to {metrics_file}")

        # Determine best model
        best_model_name = max(
            test_metrics.keys(), key=lambda k: test_metrics[k].get("roc_auc", 0)
        )
        best_model = trained_models[best_model_name]
        best_roc_auc = test_metrics[best_model_name].get("roc_auc", 0)

        # Save best model separately
        trainer.save_model(
            model=best_model,
            model_name="best_model",
            output_dir=output_dir,
            params=all_metrics[best_model_name],
            metrics=test_metrics[best_model_name],
            feature_names=feature_names,
        )

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Test ROC-AUC: {best_roc_auc:.4f}")
        logger.info(f"Test Accuracy: {test_metrics[best_model_name]['accuracy']:.4f}")
        logger.info(f"Test F1 Score: {test_metrics[best_model_name]['f1']:.4f}")
        logger.info(f"\nModels saved to: {output_dir}")
        logger.info(
            f"Evaluation report: {output_dir / 'evaluation' / 'evaluation_report.md'}"
        )
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("TRAINING FAILED!")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

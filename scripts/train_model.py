#!/usr/bin/env python3
"""
Main training script for the outage prediction model.

This script orchestrates the entire ML pipeline:
1. Load and preprocess data
2. Engineer features
3. Train models with hyperparameter optimization
4. Evaluate performance
5. Save results and visualizations
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
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


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train outage prediction model")
    parser.add_argument(
        "--data-path", type=str, default=Config.DATA_PATH, help="Path to raw data file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters for geographic grouping",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info("Starting outage prediction model training pipeline")

    try:
        # Step 1: Load and preprocess data
        logger.info("Loading raw data...")
        raw_data = load_raw_data(args.data_path)

        logger.info("Preprocessing data...")
        processed_data = preprocess_data(raw_data)

        # Step 2: Feature engineering
        logger.info("Creating temporal features...")
        data_with_temporal = create_temporal_features(processed_data)

        logger.info("Performing geographic clustering...")
        data_with_clusters, cluster_metadata = perform_geographic_clustering(
            data_with_temporal, n_clusters=args.n_clusters
        )

        logger.info("Creating aggregation features...")
        data_with_agg = create_aggregation_features(data_with_clusters)

        logger.info("Creating target variable...")
        final_data = create_target_variable(data_with_agg)

        # Remove rows with missing target
        final_data = final_data.dropna(subset=["target"])

        logger.info(f"Final dataset shape: {final_data.shape}")
        logger.info(
            f"Target distribution: {final_data['target'].value_counts().to_dict()}"
        )

        # Step 3: Train models
        logger.info("Training baseline models...")

        # Split features and target
        feature_cols = [
            col
            for col in final_data.columns
            if col
            not in [
                "target",
                "datetime",
                "isoDate",
                "latitude",
                "longitude",
                "coordinates",
                "_id",
                "location",
            ]
        ]

        X = final_data[feature_cols].fillna(final_data[feature_cols].mean())
        y = final_data["target"]

        # Split data temporally
        final_data_sorted = final_data.sort_values("datetime")
        split_idx = int(len(final_data_sorted) * 0.8)

        train_idx = final_data_sorted.index[:split_idx]
        test_idx = final_data_sorted.index[split_idx:]

        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Train models
        results = train_baseline_models(X_train, X_test, y_train, y_test)

        # Step 4: Save results
        logger.info("Saving results...")

        # Save best model
        best_model_name = max(
            results.keys(), key=lambda k: results[k]["metrics"]["roc_auc_score"]
        )
        best_model = results[best_model_name]["model"]

        model_path = output_dir / "best_model.joblib"
        best_model.save_model(model_path)

        # Save all metrics
        metrics_path = output_dir / "metrics.json"
        all_metrics = {}
        for name, result in results.items():
            # Convert numpy types for JSON serialization
            metrics = result["metrics"].copy()
            if "feature_importance" in metrics:
                metrics["feature_importance"] = dict(metrics["feature_importance"])
            all_metrics[name] = metrics

        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)

        # Save cluster metadata
        cluster_path = output_dir / "cluster_metadata.json"
        with open(cluster_path, "w") as f:
            json.dump(cluster_metadata, f, indent=2, default=str)

        # Save processed data sample
        sample_path = output_dir / "processed_data_sample.csv"
        final_data.head(1000).to_csv(sample_path, index=False)

        logger.info(f"Training completed successfully!")
        logger.info(
            f"Best model: {best_model_name} (ROC-AUC: {results[best_model_name]['metrics']['roc_auc_score']:.3f})"
        )
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()

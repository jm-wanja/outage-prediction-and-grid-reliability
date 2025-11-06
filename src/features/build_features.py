"""
Feature engineering functions for outage prediction.

This module contains functions to create features from raw outage data,
including temporal features, spatial features, and aggregation features.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


def create_temporal_features(
    df: pd.DataFrame, date_col: str = "datetime"
) -> pd.DataFrame:
    """
    Create temporal features from datetime column.

    Args:
        df: Input dataframe with datetime column
        date_col: Name of datetime column

    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()

    # Ensure datetime column is properly formatted
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract temporal features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["week_of_year"] = df[date_col].dt.isocalendar().week
    df["quarter"] = df[date_col].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["season"] = df["month"].map(
        {
            12: "dry",
            1: "dry",
            2: "dry",  # Dry season
            3: "long_rains",
            4: "long_rains",
            5: "long_rains",  # Long rains
            6: "dry",
            7: "dry",
            8: "dry",  # Dry season
            9: "short_rains",
            10: "short_rains",
            11: "short_rains",  # Short rains
        }
    )

    logger.info(f"Created temporal features for {len(df)} records")
    return df


def perform_geographic_clustering(
    df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    method: str = "kmeans",
) -> Tuple[pd.DataFrame, dict]:
    """
    Perform geographic clustering to create transformer proxies.

    Args:
        df: Input dataframe with coordinates
        n_clusters: Number of clusters (if None, will be optimized)
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        method: Clustering method ('kmeans' or 'dbscan')

    Returns:
        DataFrame with cluster assignments and clustering metadata
    """
    df = df.copy()

    # Extract coordinates
    coords = df[[lat_col, lon_col]].dropna()

    if method == "kmeans":
        if n_clusters is None:
            n_clusters = optimize_cluster_number(coords)

        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(coords)

        # Calculate cluster centers
        cluster_centers = clusterer.cluster_centers_

    elif method == "dbscan":
        # Use DBSCAN for density-based clustering
        clusterer = DBSCAN(eps=0.01, min_samples=5)  # Adjust parameters as needed
        cluster_labels = clusterer.fit_predict(coords)

        # Calculate cluster centers for valid clusters
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise cluster

        cluster_centers = []
        for label in unique_labels:
            cluster_points = coords[cluster_labels == label]
            center = cluster_points.mean().values
            cluster_centers.append(center)

        cluster_centers = np.array(cluster_centers)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Assign clusters to original dataframe
    df.loc[coords.index, "cluster_id"] = cluster_labels

    # Fill missing clusters with -1 (outliers/noise)
    df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)

    metadata = {
        "method": method,
        "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
        "cluster_centers": cluster_centers,
        "silhouette_score": (
            silhouette_score(coords, cluster_labels)
            if len(set(cluster_labels)) > 1
            else -1
        ),
    }

    logger.info(f"Created {metadata['n_clusters']} clusters using {method}")
    logger.info(f"Silhouette score: {metadata['silhouette_score']:.3f}")

    return df, metadata


def optimize_cluster_number(coords: pd.DataFrame, max_clusters: int = 20) -> int:
    """
    Optimize the number of clusters using elbow method and silhouette score.

    Args:
        coords: Coordinate data
        max_clusters: Maximum number of clusters to try

    Returns:
        Optimal number of clusters
    """
    inertias = []
    silhouette_scores = []
    k_range = range(2, min(max_clusters + 1, len(coords)))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(coords)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(coords, cluster_labels))

    # Find elbow point (simplified)
    # Choose k with highest silhouette score
    best_k = k_range[np.argmax(silhouette_scores)]

    logger.info(f"Optimal number of clusters: {best_k}")
    logger.info(f"Best silhouette score: {max(silhouette_scores):.3f}")

    return best_k


def create_aggregation_features(
    df: pd.DataFrame,
    group_cols: list = ["cluster_id"],
    time_col: str = "datetime",
    window_size: str = "7D",
) -> pd.DataFrame:
    """
    Create aggregation features by cluster and time window.

    Args:
        df: Input dataframe
        group_cols: Columns to group by
        time_col: Time column for windowing
        window_size: Size of rolling window (e.g., '7D', '30D')

    Returns:
        DataFrame with aggregation features
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Set datetime as index for rolling operations
    df_indexed = df.set_index(time_col)

    features = []

    for group_name, group_df in df_indexed.groupby(group_cols):
        group_features = group_df.copy()

        # Rolling aggregations
        rolling = group_df.rolling(window=window_size, min_periods=1)

        group_features["outage_count_7d"] = rolling.size()
        group_features["outage_rate_7d"] = group_features["outage_count_7d"] / 7

        # Lag features
        group_features["outages_last_week"] = group_features["outage_count_7d"].shift(1)
        group_features["outages_last_month"] = (
            group_df.rolling(window="30D").size().shift(1)
        )

        # Time since last outage
        group_features["days_since_last_outage"] = (
            group_features.index.to_series().diff().dt.days.fillna(0)
        )

        features.append(group_features)

    result = pd.concat(features).reset_index()

    logger.info(f"Created aggregation features for {len(result)} records")
    return result


def create_target_variable(
    df: pd.DataFrame,
    target_type: str = "outage_next_week",
    group_cols: list = ["cluster_id"],
    time_col: str = "datetime",
) -> pd.DataFrame:
    """
    Create target variable for prediction.

    Args:
        df: Input dataframe
        target_type: Type of target ('outage_next_week', 'major_outage', etc.)
        group_cols: Columns to group by
        time_col: Time column

    Returns:
        DataFrame with target variable
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if target_type == "outage_next_week":
        # Create target: will there be an outage in the next 7 days?
        df_sorted = df.sort_values([*group_cols, time_col])

        # For each group, check if there's an outage in the next week
        targets = []
        for group_name, group_df in df_sorted.groupby(group_cols):
            group_target = []

            for i, row in group_df.iterrows():
                current_date = row[time_col]
                next_week = current_date + pd.Timedelta(days=7)

                # Check if there are any outages in the next week for this cluster
                future_outages = group_df[
                    (group_df[time_col] > current_date)
                    & (group_df[time_col] <= next_week)
                ]

                target_value = 1 if len(future_outages) > 0 else 0
                group_target.append(target_value)

            group_df_copy = group_df.copy()
            group_df_copy["target"] = group_target
            targets.append(group_df_copy)

        result = pd.concat(targets)

    else:
        raise ValueError(f"Unknown target type: {target_type}")

    logger.info(f"Created target variable '{target_type}' for {len(result)} records")
    logger.info(f"Target distribution: {result['target'].value_counts().to_dict()}")

    return result

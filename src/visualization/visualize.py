"""
Visualization functions for outage prediction analysis.

This module contains functions to create various plots and visualizations
for exploratory data analysis, model results, and interactive dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_outage_timeline(
    df: pd.DataFrame,
    date_col: str = "datetime",
    figsize: Tuple[int, int] = (15, 6),
    save_path: Optional[str] = None,
):
    """
    Plot outages over time.

    Args:
        df: Dataframe with outage data
        date_col: Name of date column
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Daily outage counts
    daily_counts = df.groupby(df[date_col].dt.date).size()
    axes[0].plot(daily_counts.index, daily_counts.values, linewidth=1, alpha=0.7)
    axes[0].set_title("Daily Outage Counts", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Outages")
    axes[0].grid(True, alpha=0.3)

    # Monthly aggregation
    monthly_counts = df.groupby(df[date_col].dt.to_period("M")).size()
    monthly_counts.index = monthly_counts.index.to_timestamp()
    axes[1].bar(monthly_counts.index, monthly_counts.values, alpha=0.7, width=20)
    axes[1].set_title("Monthly Outage Counts", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Number of Outages")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved timeline plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_seasonal_patterns(
    df: pd.DataFrame,
    date_col: str = "datetime",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
):
    """
    Plot seasonal patterns in outages.

    Args:
        df: Dataframe with outage data
        date_col: Name of date column
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    df[date_col] = pd.to_datetime(df[date_col])

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Day of week patterns
    dow_counts = df.groupby(df[date_col].dt.day_name()).size()
    dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    dow_counts = dow_counts.reindex(dow_order)

    axes[0, 0].bar(dow_counts.index, dow_counts.values, alpha=0.7)
    axes[0, 0].set_title("Outages by Day of Week", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Number of Outages")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Hour of day patterns (if available)
    if df[date_col].dt.hour.nunique() > 1:
        hour_counts = df.groupby(df[date_col].dt.hour).size()
        axes[0, 1].plot(hour_counts.index, hour_counts.values, marker="o", alpha=0.7)
        axes[0, 1].set_title("Outages by Hour of Day", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Number of Outages")
        axes[0, 1].set_xlabel("Hour")
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "Hour data not available",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("Outages by Hour of Day", fontsize=12, fontweight="bold")

    # Monthly patterns
    month_counts = df.groupby(df[date_col].dt.month).size()
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    axes[1, 0].bar(
        range(1, 13), [month_counts.get(i, 0) for i in range(1, 13)], alpha=0.7
    )
    axes[1, 0].set_title("Outages by Month", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Number of Outages")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(month_names, rotation=45)

    # Quarterly patterns
    quarter_counts = df.groupby(df[date_col].dt.quarter).size()
    axes[1, 1].bar(quarter_counts.index, quarter_counts.values, alpha=0.7)
    axes[1, 1].set_title("Outages by Quarter", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Number of Outages")
    axes[1, 1].set_xlabel("Quarter")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved seasonal patterns plot to {save_path}")
        plt.close()
    else:
        plt.show()


def create_outage_map(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    cluster_col: Optional[str] = None,
    center_lat: float = -1.2921,  # Nairobi coordinates
    center_lon: float = 36.8219,
    zoom_start: int = 7,
) -> folium.Map:
    """
    Create an interactive map of outages.

    Args:
        df: Dataframe with outage data
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        cluster_col: Name of cluster column (optional)
        center_lat: Map center latitude
        center_lon: Map center longitude
        zoom_start: Initial zoom level

    Returns:
        Folium map object
    """
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap"
    )

    # Filter out missing coordinates
    df_clean = df.dropna(subset=[lat_col, lon_col])

    if cluster_col and cluster_col in df_clean.columns:
        # Color by cluster
        colors = [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "darkred",
            "lightred",
            "beige",
            "darkblue",
            "darkgreen",
            "cadetblue",
            "darkpurple",
            "white",
            "pink",
            "lightblue",
            "lightgreen",
            "gray",
            "black",
            "lightgray",
        ]

        unique_clusters = df_clean[cluster_col].unique()
        cluster_colors = {
            cluster: colors[i % len(colors)]
            for i, cluster in enumerate(unique_clusters)
        }

        for cluster in unique_clusters:
            if cluster == -1:  # Noise/outlier cluster
                continue

            cluster_data = df_clean[df_clean[cluster_col] == cluster]

            for idx, row in cluster_data.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    popup=f'Cluster: {cluster}<br>Date: {row.get("datetime", "N/A")}',
                    color=cluster_colors[cluster],
                    fill=True,
                    fillColor=cluster_colors[cluster],
                ).add_to(m)
    else:
        # Add all points
        marker_cluster = MarkerCluster().add_to(m)

        for idx, row in df_clean.iterrows():
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=f'Outage: {row.get("datetime", "N/A")}',
            ).add_to(marker_cluster)

    # Add heat map layer
    heat_data = [[row[lat_col], row[lon_col]] for idx, row in df_clean.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m)

    logger.info(f"Created map with {len(df_clean)} outage locations")

    return m


def plot_cluster_analysis(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None,
):
    """
    Plot cluster analysis results.

    Args:
        df: Dataframe with cluster assignments
        cluster_col: Name of cluster column
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Cluster size distribution
    cluster_counts = df[cluster_col].value_counts().sort_index()
    axes[0, 0].bar(cluster_counts.index, cluster_counts.values, alpha=0.7)
    axes[0, 0].set_title("Cluster Size Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Cluster ID")
    axes[0, 0].set_ylabel("Number of Outages")

    # Temporal distribution by cluster
    if "datetime" in df.columns:
        df["month"] = pd.to_datetime(df["datetime"]).dt.month
        cluster_temporal = (
            df.groupby([cluster_col, "month"]).size().unstack(fill_value=0)
        )

        im = axes[0, 1].imshow(cluster_temporal.values, aspect="auto", cmap="YlOrRd")
        axes[0, 1].set_title(
            "Outages by Cluster and Month", fontsize=12, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Month")
        axes[0, 1].set_ylabel("Cluster ID")
        axes[0, 1].set_yticks(range(len(cluster_temporal.index)))
        axes[0, 1].set_yticklabels(cluster_temporal.index)
        plt.colorbar(im, ax=axes[0, 1])

    # Geographic distribution
    if "latitude" in df.columns and "longitude" in df.columns:
        for cluster_id in df[cluster_col].unique()[:10]:  # Limit to first 10 clusters
            if cluster_id == -1:
                continue
            cluster_data = df[df[cluster_col] == cluster_id]
            axes[1, 0].scatter(
                cluster_data["longitude"],
                cluster_data["latitude"],
                label=f"Cluster {cluster_id}",
                alpha=0.6,
                s=20,
            )
        axes[1, 0].set_title(
            "Geographic Distribution of Clusters", fontsize=12, fontweight="bold"
        )
        axes[1, 0].set_xlabel("Longitude")
        axes[1, 0].set_ylabel("Latitude")
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Cluster statistics
    cluster_stats = (
        df.groupby(cluster_col).agg({"latitude": "mean", "longitude": "mean"}).round(4)
    )

    axes[1, 1].axis("tight")
    axes[1, 1].axis("off")
    table_data = cluster_stats.head(10).values
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=["Mean Lat", "Mean Lon"],
        rowLabels=[f"Cluster {i}" for i in cluster_stats.head(10).index],
        cellLoc="center",
        loc="center",
    )
    axes[1, 1].set_title("Cluster Centers (Top 10)", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved cluster analysis plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_model_performance(metrics: Dict, figsize: Tuple[int, int] = (15, 10)):
    """
    Plot model performance metrics.

    Args:
        metrics: Dictionary of model metrics
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ROC-AUC comparison (if multiple models)
    if isinstance(metrics, dict) and "roc_auc_score" not in metrics:
        model_names = list(metrics.keys())
        roc_scores = [
            metrics[model]["metrics"]["roc_auc_score"] for model in model_names
        ]

        axes[0, 0].bar(model_names, roc_scores, alpha=0.7)
        axes[0, 0].set_title("Model ROC-AUC Comparison")
        axes[0, 0].set_ylabel("ROC-AUC Score")
        axes[0, 0].set_ylim(0, 1)

        # Feature importance (for best model)
        best_model = max(
            model_names, key=lambda k: metrics[k]["metrics"]["roc_auc_score"]
        )
        if "feature_importance" in metrics[best_model]["metrics"]:
            feat_imp = metrics[best_model]["metrics"]["feature_importance"]
            top_features = dict(list(feat_imp.items())[:10])

            axes[0, 1].barh(
                list(top_features.keys()), list(top_features.values()), alpha=0.7
            )
            axes[0, 1].set_title(f"Top 10 Features ({best_model})")
            axes[0, 1].set_xlabel("Importance")

        # Confusion matrix (for best model)
        conf_matrix = np.array(metrics[best_model]["metrics"]["confusion_matrix"])
        im = axes[1, 0].imshow(conf_matrix, cmap="Blues")
        axes[1, 0].set_title(f"Confusion Matrix ({best_model})")
        axes[1, 0].set_xlabel("Predicted")
        axes[1, 0].set_ylabel("Actual")

        # Add text annotations
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                axes[1, 0].text(j, i, conf_matrix[i, j], ha="center", va="center")

        # Precision/Recall by class
        class_report = metrics[best_model]["metrics"]["classification_report"]
        classes = [k for k in class_report.keys() if k.isdigit()]
        precision = [class_report[c]["precision"] for c in classes]
        recall = [class_report[c]["recall"] for c in classes]

        x = np.arange(len(classes))
        width = 0.35

        axes[1, 1].bar(x - width / 2, precision, width, label="Precision", alpha=0.7)
        axes[1, 1].bar(x + width / 2, recall, width, label="Recall", alpha=0.7)
        axes[1, 1].set_title("Precision and Recall by Class")
        axes[1, 1].set_xlabel("Class")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(classes)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def create_interactive_dashboard(
    df: pd.DataFrame, predictions: Optional[pd.DataFrame] = None
):
    """
    Create an interactive dashboard using Plotly.

    Args:
        df: Main dataframe with outage data
        predictions: Optional dataframe with model predictions
    """
    # Create subplot structure
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "Outages Over Time",
            "Geographic Distribution",
            "Seasonal Patterns",
            "Risk Prediction (if available)",
            "Cluster Analysis",
            "Model Performance",
        ],
        specs=[
            [{"secondary_y": False}, {"type": "mapbox"}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Time series plot
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        daily_counts = df.groupby(df["datetime"].dt.date).size().reset_index()
        daily_counts.columns = ["date", "count"]

        fig.add_trace(
            go.Scatter(
                x=daily_counts["date"], y=daily_counts["count"], name="Daily Outages"
            ),
            row=1,
            col=1,
        )

    # Geographic scatter plot (simplified for subplot)
    if "latitude" in df.columns and "longitude" in df.columns:
        df_clean = df.dropna(subset=["latitude", "longitude"])

        fig.add_trace(
            go.Scattermapbox(
                lat=df_clean["latitude"],
                lon=df_clean["longitude"],
                mode="markers",
                marker=dict(size=8, color="red", opacity=0.6),
                name="Outages",
                text=df_clean.get("datetime", "N/A"),
            ),
            row=1,
            col=2,
        )

    # Set mapbox style
    fig.update_layout(
        mapbox=dict(
            style="open-street-map", center=dict(lat=-1.2921, lon=36.8219), zoom=6
        )
    )

    # Update layout
    fig.update_layout(
        height=1000, title_text="Outage Prediction Dashboard", showlegend=True
    )

    return fig

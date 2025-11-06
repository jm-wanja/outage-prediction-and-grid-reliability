#!/usr/bin/env python3
"""
Exploratory Data Analysis script.

This script performs comprehensive EDA on the KPLC outage data
and generates various visualizations and reports.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import (
    create_temporal_features,
    perform_geographic_clustering,
)
from src.visualization.visualize import (
    plot_outage_timeline,
    plot_seasonal_patterns,
    create_outage_map,
    plot_cluster_analysis,
    create_interactive_dashboard,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main EDA pipeline."""
    parser = argparse.ArgumentParser(description="Perform Exploratory Data Analysis")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/kplc_interruption_data.json",
        help="Path to raw data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Generate interactive plots"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Exploratory Data Analysis")

    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        raw_data = load_raw_data(args.data_path)

        # Convert list to DataFrame
        df = pd.DataFrame(raw_data)

        # Preprocess the DataFrame
        processed_data = preprocess_data(df)

        logger.info(f"Dataset shape: {processed_data.shape}")
        logger.info(
            f"Date range: {processed_data['datetime'].min()} to {processed_data['datetime'].max()}"
        )

        # Add temporal features for analysis
        data_with_temporal = create_temporal_features(processed_data)

        # Perform clustering for geographic analysis
        data_with_clusters, _ = perform_geographic_clustering(data_with_temporal)

        # Generate basic statistics
        logger.info("Generating basic statistics...")
        stats_report = generate_basic_stats(data_with_clusters)

        # Save statistics report
        stats_path = output_dir / "basic_statistics.txt"
        with open(stats_path, "w") as f:
            f.write(stats_report)

        # Generate visualizations
        logger.info("Generating visualizations...")

        # Timeline plots
        logger.info("Creating timeline plots...")
        timeline_path = output_dir / "timeline_analysis.png"
        plot_outage_timeline(data_with_clusters, save_path=str(timeline_path))

        # Seasonal patterns
        logger.info("Creating seasonal pattern plots...")
        seasonal_path = output_dir / "seasonal_patterns.png"
        plot_seasonal_patterns(data_with_clusters, save_path=str(seasonal_path))

        # Cluster analysis
        logger.info("Creating cluster analysis plots...")
        cluster_path = output_dir / "cluster_analysis.png"
        plot_cluster_analysis(data_with_clusters, save_path=str(cluster_path))

        # Interactive map
        if args.interactive:
            logger.info("Creating interactive map...")
            outage_map = create_outage_map(data_with_clusters, cluster_col="cluster_id")
            map_path = output_dir / "outage_map.html"
            outage_map.save(str(map_path))
            logger.info(f"Interactive map saved to: {map_path}")

            # Interactive dashboard
            logger.info("Creating interactive dashboard...")
            dashboard = create_interactive_dashboard(data_with_clusters)
            dashboard_path = output_dir / "dashboard.html"
            dashboard.write_html(str(dashboard_path))
            logger.info(f"Interactive dashboard saved to: {dashboard_path}")

        # Generate summary report
        logger.info("Generating summary report...")
        summary_report = generate_summary_report(data_with_clusters)
        summary_path = output_dir / "eda_summary.md"
        with open(summary_path, "w") as f:
            f.write(summary_report)

        logger.info("EDA completed successfully!")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"EDA failed with error: {str(e)}")
        raise


def generate_basic_stats(df):
    """Generate basic statistics report."""
    report = []
    report.append("# Basic Statistics Report")
    report.append("=" * 50)
    report.append("")

    # Dataset overview
    report.append(f"Dataset Shape: {df.shape}")
    report.append(f"Total Outages: {len(df)}")
    report.append("")

    # Temporal statistics
    report.append("## Temporal Statistics")
    report.append(f"Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
    report.append(
        f"Duration: {(df['datetime'].max() - df['datetime'].min()).days} days"
    )
    report.append("")

    # Geographic statistics
    if "latitude" in df.columns and "longitude" in df.columns:
        report.append("## Geographic Statistics")
        lat_bounds = (df["latitude"].min(), df["latitude"].max())
        lon_bounds = (df["longitude"].min(), df["longitude"].max())
        report.append(f"Latitude Range: {lat_bounds[0]:.4f} to {lat_bounds[1]:.4f}")
        report.append(f"Longitude Range: {lon_bounds[0]:.4f} to {lon_bounds[1]:.4f}")
        report.append("")

    # Clustering statistics
    if "cluster_id" in df.columns:
        report.append("## Clustering Statistics")
        cluster_counts = df["cluster_id"].value_counts().sort_index()
        report.append(f"Number of Clusters: {len(cluster_counts)}")
        report.append(f"Largest Cluster: {cluster_counts.max()} outages")
        report.append(f"Smallest Cluster: {cluster_counts.min()} outages")
        report.append("")

    # Missing data
    report.append("## Missing Data")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        for col, count in missing_data.items():
            percentage = (count / len(df)) * 100
            report.append(f"{col}: {count} ({percentage:.2f}%)")
    else:
        report.append("No missing data found")

    return "\n".join(report)


def generate_summary_report(df):
    """Generate markdown summary report."""
    report = []
    report.append("# Exploratory Data Analysis Summary")
    report.append("")
    report.append("## Dataset Overview")
    report.append(f"- **Total Outages**: {len(df):,}")
    report.append(
        f"- **Date Range**: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}"
    )
    report.append(
        f"- **Duration**: {(df['datetime'].max() - df['datetime'].min()).days} days"
    )
    report.append("")

    # Temporal patterns
    report.append("## Key Findings")
    report.append("")

    # Most common day of week
    most_common_dow = df["day_of_week"].mode().iloc[0]
    most_common_dow_int = int(most_common_dow)  # Convert to integer
    dow_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    report.append(f"- **Most Common Outage Day**: {dow_names[most_common_dow_int]}")

    # Most common month
    most_common_month = df["month"].mode().iloc[0]
    most_common_month_int = (
        int(most_common_month) - 1
    )  # Convert to integer and adjust for 0-indexing
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
    report.append(
        f"- **Most Common Outage Month**: {month_names[most_common_month_int]}"
    )

    # Seasonal patterns
    season_counts = df["season"].value_counts()
    peak_season = season_counts.index[0]
    report.append(
        f"- **Peak Outage Season**: {peak_season.title()} ({season_counts.iloc[0]} outages)"
    )

    # Geographic insights
    if "cluster_id" in df.columns:
        n_clusters = df["cluster_id"].nunique()
        report.append(f"- **Geographic Clusters Identified**: {n_clusters}")

        largest_cluster = df["cluster_id"].value_counts().iloc[0]
        report.append(f"- **Largest Cluster Size**: {largest_cluster} outages")

    report.append("")
    report.append("## Recommendations")
    report.append("- Focus predictive maintenance during peak season")
    report.append("- Investigate high-frequency outage clusters")
    report.append("- Consider temporal patterns for resource allocation")

    return "\n".join(report)


if __name__ == "__main__":
    main()

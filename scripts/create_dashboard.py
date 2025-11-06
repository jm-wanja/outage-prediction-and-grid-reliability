"""
Create comprehensive interactive dashboard for outage analysis.

This script creates a rich, interactive HTML dashboard with multiple
coordinated views showing temporal, spatial, and statistical patterns
in the power outage data.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import logging
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import (
    create_temporal_features,
    perform_geographic_clustering,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Load and prepare the outage data from JSON."""
    logger.info(f"Loading data from {data_path}")

    # Load raw data from JSON
    raw_data = load_raw_data(data_path)

    # Convert to DataFrame
    df = pd.DataFrame(raw_data)

    # Preprocess
    df = preprocess_data(df)

    # Add temporal features
    df = create_temporal_features(df)

    # Perform geographic clustering
    df, _ = perform_geographic_clustering(df)

    # Extract additional temporal features for dashboard
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["month"] = df["datetime"].dt.month
    df["month_name"] = df["datetime"].dt.month_name()
    df["quarter"] = df["datetime"].dt.quarter
    df["year"] = df["datetime"].dt.year

    logger.info(f"Loaded and processed {len(df)} records")
    return df


def create_comprehensive_dashboard(df: pd.DataFrame, save_path: str):
    """
    Create a comprehensive interactive dashboard.

    Args:
        df: Dataframe with outage data
        save_path: Path to save the HTML dashboard
    """
    logger.info("Creating comprehensive dashboard...")

    # Create figure with subplots
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "📅 Daily Outage Timeline",
            "🗺️ Geographic Distribution by Cluster",
            "📊 Monthly Outage Patterns",
            "⏰ Hourly Outage Distribution",
            "📍 Outages by Cluster",
            "📈 Day of Week Patterns",
            "🔢 Quarterly Trends",
            "📋 Summary Statistics",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.12,
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    # 1. Daily Timeline (Row 1, Col 1)
    daily_counts = df.groupby("date").size().reset_index(name="count")
    daily_counts["date"] = pd.to_datetime(daily_counts["date"])

    fig.add_trace(
        go.Scatter(
            x=daily_counts["date"],
            y=daily_counts["count"],
            mode="lines+markers",
            name="Daily Outages",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br>"
            + "<b>Outages:</b> %{y}<br>"
            + "<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. Geographic Distribution (Row 1, Col 2)
    if (
        "cluster_id" in df.columns
        and "latitude" in df.columns
        and "longitude" in df.columns
    ):
        df_clean = df.dropna(subset=["latitude", "longitude", "cluster_id"])

        # Calculate cluster centers
        cluster_centers = (
            df_clean.groupby("cluster_id")
            .agg({"latitude": "mean", "longitude": "mean"})
            .reset_index()
        )
        cluster_counts = df_clean.groupby("cluster_id").size().reset_index(name="count")
        cluster_summary = cluster_centers.merge(cluster_counts, on="cluster_id")

        colors = px.colors.qualitative.Set1

        for idx, row in cluster_summary.iterrows():
            cluster_data = df_clean[df_clean["cluster_id"] == row["cluster_id"]]
            color_idx = int(row["cluster_id"]) % len(colors)

            fig.add_trace(
                go.Scatter(
                    x=cluster_data["longitude"],
                    y=cluster_data["latitude"],
                    mode="markers",
                    name=f'Cluster {int(row["cluster_id"])}',
                    marker=dict(
                        size=6,
                        color=colors[color_idx],
                        opacity=0.6,
                        line=dict(width=0.5, color="white"),
                    ),
                    hovertemplate="<b>Cluster:</b> "
                    + str(int(row["cluster_id"]))
                    + "<br>"
                    + "<b>Lat:</b> %{y:.4f}<br>"
                    + "<b>Lon:</b> %{x:.4f}<br>"
                    + "<extra></extra>",
                ),
                row=1,
                col=2,
            )

    # 3. Monthly Patterns (Row 2, Col 1)
    monthly_counts = df.groupby("month_name").size().reset_index(name="count")
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    monthly_counts["month_name"] = pd.Categorical(
        monthly_counts["month_name"], categories=month_order, ordered=True
    )
    monthly_counts = monthly_counts.sort_values("month_name")

    fig.add_trace(
        go.Bar(
            x=monthly_counts["month_name"],
            y=monthly_counts["count"],
            name="Monthly Outages",
            marker=dict(color="#ff7f0e"),
            hovertemplate="<b>Month:</b> %{x}<br>"
            + "<b>Outages:</b> %{y}<br>"
            + "<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 4. Hourly Distribution (Row 2, Col 2)
    hourly_counts = df.groupby("hour").size().reset_index(name="count")

    fig.add_trace(
        go.Bar(
            x=hourly_counts["hour"],
            y=hourly_counts["count"],
            name="Hourly Distribution",
            marker=dict(color="#2ca02c"),
            hovertemplate="<b>Hour:</b> %{x}:00<br>"
            + "<b>Outages:</b> %{y}<br>"
            + "<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # 5. Cluster Distribution (Row 3, Col 1)
    if "cluster_id" in df.columns:
        cluster_counts = df.groupby("cluster_id").size().reset_index(name="count")
        cluster_counts = cluster_counts.sort_values("cluster_id")

        fig.add_trace(
            go.Bar(
                x=cluster_counts["cluster_id"].astype(str),
                y=cluster_counts["count"],
                name="Cluster Distribution",
                marker=dict(color="#d62728"),
                hovertemplate="<b>Cluster:</b> %{x}<br>"
                + "<b>Outages:</b> %{y}<br>"
                + "<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # 6. Day of Week Patterns (Row 3, Col 2)
    dow_counts = df.groupby("day_of_week").size().reset_index(name="count")
    dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    dow_counts["day_of_week"] = pd.Categorical(
        dow_counts["day_of_week"], categories=dow_order, ordered=True
    )
    dow_counts = dow_counts.sort_values("day_of_week")

    fig.add_trace(
        go.Bar(
            x=dow_counts["day_of_week"],
            y=dow_counts["count"],
            name="Day of Week",
            marker=dict(color="#9467bd"),
            hovertemplate="<b>Day:</b> %{x}<br>"
            + "<b>Outages:</b> %{y}<br>"
            + "<extra></extra>",
        ),
        row=3,
        col=2,
    )

    # 7. Quarterly Trends (Row 4, Col 1)
    quarterly_counts = df.groupby(["year", "quarter"]).size().reset_index(name="count")
    quarterly_counts["period"] = (
        quarterly_counts["year"].astype(str)
        + "-Q"
        + quarterly_counts["quarter"].astype(str)
    )

    fig.add_trace(
        go.Bar(
            x=quarterly_counts["period"],
            y=quarterly_counts["count"],
            name="Quarterly Trends",
            marker=dict(color="#8c564b"),
            hovertemplate="<b>Period:</b> %{x}<br>"
            + "<b>Outages:</b> %{y}<br>"
            + "<extra></extra>",
        ),
        row=4,
        col=1,
    )

    # 8. Summary Statistics Table (Row 4, Col 2)
    total_outages = len(df)
    date_range = f"{df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}"
    num_clusters = df["cluster_id"].nunique() if "cluster_id" in df.columns else "N/A"
    avg_daily = df.groupby("date").size().mean()
    peak_day = df.groupby("date").size().idxmax()
    peak_count = df.groupby("date").size().max()

    # Most affected areas
    if "area" in df.columns:
        top_area = df["area"].value_counts().index[0]
        top_area_count = df["area"].value_counts().iloc[0]
        area_info = f"{top_area} ({top_area_count})"
    else:
        area_info = "N/A"

    summary_data = {
        "Metric": [
            "Total Outages",
            "Date Range",
            "Number of Clusters",
            "Avg Daily Outages",
            "Peak Day",
            "Peak Day Count",
            "Most Affected Area",
        ],
        "Value": [
            f"{total_outages:,}",
            date_range,
            str(num_clusters),
            f"{avg_daily:.1f}",
            str(peak_day),
            f"{peak_count:,}",
            area_info,
        ],
    }

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>"],
                fill_color="#1f77b4",
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[summary_data["Metric"], summary_data["Value"]],
                fill_color="#f0f0f0",
                font=dict(size=11),
                align="left",
                height=25,
            ),
        ),
        row=4,
        col=2,
    )

    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Number of Outages", row=1, col=1)

    fig.update_xaxes(title_text="Longitude", row=1, col=2)
    fig.update_yaxes(title_text="Latitude", row=1, col=2)

    fig.update_xaxes(title_text="Month", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Number of Outages", row=2, col=1)

    fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
    fig.update_yaxes(title_text="Number of Outages", row=2, col=2)

    fig.update_xaxes(title_text="Cluster ID", row=3, col=1)
    fig.update_yaxes(title_text="Number of Outages", row=3, col=1)

    fig.update_xaxes(title_text="Day of Week", row=3, col=2, tickangle=45)
    fig.update_yaxes(title_text="Number of Outages", row=3, col=2)

    fig.update_xaxes(title_text="Quarter", row=4, col=1, tickangle=45)
    fig.update_yaxes(title_text="Number of Outages", row=4, col=1)

    # Update overall layout
    fig.update_layout(
        title={
            "text": "⚡ Kenya Power Outage Analysis Dashboard<br>"
            + "<sub>Interactive visualization of power outage patterns and trends</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 24, "color": "#1f1f1f"},
        },
        height=1600,
        showlegend=False,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        hovermode="closest",
        plot_bgcolor="rgba(240,240,240,0.5)",
    )

    # Add interactivity
    fig.update_traces(marker_line_width=0.5, selector=dict(type="bar"))

    # Save to HTML
    logger.info(f"Saving dashboard to {save_path}")
    fig.write_html(
        save_path,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "outage_dashboard",
                "height": 1600,
                "width": 1200,
                "scale": 2,
            },
        },
    )

    logger.info("Dashboard created successfully!")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 DASHBOARD SUMMARY")
    print("=" * 60)
    print(f"Total Outages: {total_outages:,}")
    print(f"Date Range: {date_range}")
    print(f"Clusters: {num_clusters}")
    print(f"Average Daily Outages: {avg_daily:.1f}")
    print(f"Peak Day: {peak_day} ({peak_count:,} outages)")
    print(f"\nDashboard saved to: {save_path}")
    print("=" * 60 + "\n")


def main():
    """Main execution function."""
    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "kplc_interruption_data.json"
    save_path = project_root / "reports" / "figures" / "dashboard.html"

    # Check if data exists
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please check the data path.")
        return

    # Load data
    df = load_and_prepare_data(str(data_path))

    # Create dashboard
    create_comprehensive_dashboard(df, str(save_path))


if __name__ == "__main__":
    main()

# Data Pipeline

This section documents the data flow and processing steps in the Outage Prediction and Grid Reliability project.

## Data Sources

- **KPLC Electricity Interruption Data**: [Kaggle Dataset](https://www.kaggle.com/datasets/kingrobi/kplc-electricity-interruption-data-kenya)
- **File location**: `data/kplc_interruption_data.json`

## Data Structure

- **latitude, longitude**: Outage location
- **datetime**: Timestamp of outage
- **area**: (if available) Area or region name
- **Other fields**: As provided in the raw data

## Preprocessing Steps

- Remove duplicates and missing values
- Parse and standardize datetime
- Extract and clean coordinates
- Filter out invalid or outlier records

## Feature Engineering

- **Temporal features**: Year, month, day of week, hour, season
- **Spatial features**: Cluster ID, geographic density
- **Historical features**: Rolling outage counts, lag features
- **External features**: (Planned) Weather, holidays

## Clustering

- **Purpose**: Group outages into zones (proxy for transformers)
- **Methods**: K-means (default), DBSCAN (optional)
- **Output**: `cluster_id` column in processed data

## Data Flow Diagram

```
Raw JSON → Preprocessing → Feature Engineering → Clustering → Model Training
```

## Example Usage

```python
from src.data.load_data import load_raw_data, preprocess_data
from src.features.build_features import create_temporal_features, perform_geographic_clustering

raw_data = load_raw_data('data/kplc_interruption_data.json')
df = preprocess_data(raw_data)
df = create_temporal_features(df)
df, _ = perform_geographic_clustering(df)
```

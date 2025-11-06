# API Reference

## Data Module

### `src.data.load_data`

#### Functions

##### `load_raw_data(file_path: str) -> pd.DataFrame`

Load raw KPLC outage data from JSON file.

**Parameters:**

- `file_path`: Path to the JSON data file

**Returns:**

- DataFrame with raw outage data

**Raises:**

- `FileNotFoundError`: If the data file doesn't exist

##### `preprocess_data(df: pd.DataFrame) -> pd.DataFrame`

Preprocess raw outage data for analysis.

**Parameters:**

- `df`: Raw dataframe from load_raw_data

**Returns:**

- Preprocessed DataFrame with cleaned coordinates and datetime

## Features Module

### `src.features.build_features`

#### Functions

##### `create_temporal_features(df: pd.DataFrame, date_col: str = 'datetime') -> pd.DataFrame`

Create temporal features from datetime column.

**Parameters:**

- `df`: Input dataframe with datetime column
- `date_col`: Name of datetime column (default: 'datetime')

**Returns:**

- DataFrame with additional temporal features (year, month, day_of_week, season, etc.)

##### `perform_geographic_clustering(df: pd.DataFrame, n_clusters: Optional[int] = None, method: str = 'kmeans') -> Tuple[pd.DataFrame, dict]`

Perform geographic clustering to create transformer proxies.

**Parameters:**

- `df`: Input dataframe with coordinates
- `n_clusters`: Number of clusters (if None, will be optimized)
- `method`: Clustering method ('kmeans' or 'dbscan')

**Returns:**

- Tuple of (DataFrame with cluster assignments, clustering metadata)

## Models Module

### `src.models.train_model`

#### Classes

##### `OutagePredictionModel`

Main model class for outage prediction.

###### Methods

**`**init**(model_type: str = 'random_forest', **kwargs)`\*\*
Initialize the model.

**`train(X_train: pd.DataFrame, y_train: pd.Series)`**
Train the model on provided data.

**`predict(X: pd.DataFrame) -> np.ndarray`**
Make predictions on new data.

**`evaluate(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]`**
Evaluate model performance and return metrics.

## Visualization Module

### `src.visualization.visualize`

#### Functions

##### `plot_outage_timeline(df: pd.DataFrame, date_col: str = 'datetime')`

Plot outages over time with daily and monthly aggregations.

##### `create_outage_map(df: pd.DataFrame, lat_col: str = 'latitude', lon_col: str = 'longitude') -> folium.Map`

Create an interactive map of outage locations.

##### `plot_model_performance(metrics: Dict)`

Plot comprehensive model performance metrics.

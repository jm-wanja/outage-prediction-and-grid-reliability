# Tutorials

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jm-wanja/outage-prediction-and-grid-reliability.git
cd outage-prediction-and-grid-reliability

# Create conda environment
conda env create -f environment.yml
conda activate outage-prediction

# Or use pip
pip install -r requirements.txt
```

### 2. Quick Start

#### Load and Explore Data

```python
from src.data.load_data import load_raw_data, preprocess_data
from src.visualization.visualize import plot_outage_timeline

# Load data
raw_data = load_raw_data('data/kplc_interruption_data.json')
processed_data = preprocess_data(raw_data)

# Basic exploration
print(f"Dataset shape: {processed_data.shape}")
print(f"Date range: {processed_data['datetime'].min()} to {processed_data['datetime'].max()}")

# Visualize timeline
plot_outage_timeline(processed_data)
```

#### Feature Engineering

```python
from src.features.build_features import (
    create_temporal_features,
    perform_geographic_clustering
)

# Create temporal features
data_with_features = create_temporal_features(processed_data)

# Perform geographic clustering
clustered_data, metadata = perform_geographic_clustering(data_with_features)
print(f"Created {metadata['n_clusters']} clusters")
```

#### Train Model

```python
from src.models.train_model import OutagePredictionModel
from src.features.build_features import create_target_variable

# Create target variable
final_data = create_target_variable(clustered_data)

# Initialize and train model
model = OutagePredictionModel('random_forest')
X_train, X_test, y_train, y_test = model.prepare_data(final_data)
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"ROC-AUC Score: {metrics['roc_auc_score']:.3f}")
```

## Advanced Usage

### Custom Feature Engineering

```python
import pandas as pd
from src.features.build_features import create_aggregation_features

# Create custom aggregation features
data_with_agg = create_aggregation_features(
    clustered_data,
    group_cols=['cluster_id'],
    window_size='14D'  # 14-day window
)
```

### Model Hyperparameter Tuning

```python
# Custom parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Tune hyperparameters
tuning_results = model.tune_hyperparameters(X_train, y_train, param_grid)
print(f"Best parameters: {tuning_results['best_params']}")
```

### Interactive Visualizations

```python
from src.visualization.visualize import create_outage_map, create_interactive_dashboard

# Create interactive map
outage_map = create_outage_map(clustered_data, cluster_col='cluster_id')
outage_map.save('outage_map.html')

# Create dashboard
dashboard = create_interactive_dashboard(clustered_data)
dashboard.write_html('dashboard.html')
```

## Command Line Interface

### Run Complete Training Pipeline

```bash
python scripts/train_model.py --data-path data/kplc_interruption_data.json --output-dir models
```

### Perform Exploratory Data Analysis

```bash
python scripts/run_eda.py --data-path data/kplc_interruption_data.json --interactive
```

## Working with Jupyter Notebooks

Start with the provided notebooks:

1. **01_exploratory_data_analysis.ipynb** - Comprehensive EDA
2. **02_feature_engineering.ipynb** - Feature creation and selection
3. **03_model_training.ipynb** - Model training and evaluation
4. **04_model_interpretation.ipynb** - Model interpretation and insights

```bash
# Start Jupyter
jupyter notebook notebooks/
```

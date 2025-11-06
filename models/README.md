# Model Training Pipeline

Comprehensive machine learning pipeline for predicting power outages using KPLC interruption data.

## Overview

This pipeline trains multiple classification models to predict whether an outage will occur in a specific geographic cluster within the next 7 days.

## Features

- **Multiple Model Types**: Random Forest, Gradient Boosting, Logistic Regression, XGBoost, LightGBM
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Temporal Data Splitting**: Train/validation/test splits respect time ordering
- **Geographic Clustering**: K-means clustering for spatial features
- **Comprehensive Evaluation**: ROC curves, confusion matrices, feature importance
- **Production-Ready**: Model serialization, metadata tracking, and evaluation reports

## Quick Start

### 1. Train Baseline Models (Fast)

Train all models with default parameters:

```bash
python scripts/train_model_new.py --data-path data/kplc_interruption_data.json
```

### 2. Train Specific Model with Optimization

Train XGBoost with hyperparameter tuning:

```bash
python scripts/train_model_new.py \
  --model-type xgboost \
  --optimize \
  --n-trials 100
```

### 3. Train All Models with Optimization

Train and optimize all available models:

```bash
python scripts/train_model_new.py \
  --model-type all \
  --optimize \
  --n-trials 50 \
  --output-dir models/optimized
```

## Command-Line Arguments

| Argument         | Default                            | Description                                                                                           |
| ---------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `--data-path`    | `data/kplc_interruption_data.json` | Path to raw data file                                                                                 |
| `--output-dir`   | `models`                           | Directory to save trained models                                                                      |
| `--model-type`   | `all`                              | Model type: `all`, `random_forest`, `gradient_boosting`, `logistic_regression`, `xgboost`, `lightgbm` |
| `--optimize`     | `False`                            | Enable hyperparameter optimization                                                                    |
| `--n-trials`     | `50`                               | Number of Optuna optimization trials                                                                  |
| `--n-clusters`   | `None`                             | Number of geographic clusters (auto-detect if not specified)                                          |
| `--test-size`    | `0.2`                              | Proportion of data for test set                                                                       |
| `--val-size`     | `0.2`                              | Proportion of training data for validation                                                            |
| `--random-state` | `42`                               | Random seed for reproducibility                                                                       |

## Pipeline Steps

### 1. Data Loading & Preprocessing

- Load raw JSON data
- Extract coordinates from nested structures
- Clean invalid geographic coordinates
- Convert datetime formats

### 2. Feature Engineering

- **Temporal Features**: Year, month, day of week, quarter, season, day of year
- **Geographic Features**: Latitude, longitude, cluster assignment
- **Clustering**: K-means geographic clustering with silhouette score optimization

### 3. Target Variable Creation

- Binary classification: Will there be an outage in the next 7 days?
- Computed per geographic cluster
- Handles class imbalance

### 4. Data Splitting

- **Temporal Split**: Train/validation/test respect time ordering
- **Stratified Sampling**: Maintains class balance in validation set
- Default: 60% train, 20% validation, 20% test

### 5. Model Training

- **Baseline**: Default hyperparameters
- **Optimized**: Optuna-based hyperparameter search
- Cross-validation for robust evaluation

### 6. Model Evaluation

- Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision
- Confusion matrices
- ROC and Precision-Recall curves
- Feature importance analysis
- Markdown evaluation report

## Output Structure

```
models/
├── best_model.joblib                      # Best performing model
├── best_model_metadata.json               # Model metadata
├── all_metrics.json                       # All training and test metrics
├── random_forest_baseline.joblib          # Individual model files
├── random_forest_baseline_metadata.json
├── xgboost_optimized.joblib
├── xgboost_optimized_metadata.json
└── evaluation/
    ├── evaluation_report.md               # Markdown report
    ├── model_comparison.csv               # Model comparison table
    ├── detailed_metrics.json              # Detailed metrics
    ├── roc_curves.png                     # ROC curves comparison
    ├── precision_recall_curves.png        # PR curves comparison
    ├── confusion_matrix_xgboost_optimized.png
    ├── confusion_matrix_random_forest_baseline.png
    └── feature_importance_*.png           # Feature importance plots
```

## Model Classes

### ModelTrainer

Main class for training models with hyperparameter optimization.

```python
from src.models import ModelTrainer

trainer = ModelTrainer(random_state=42)

# Train baseline model
model, metrics = trainer.train_baseline(
    model_type='random_forest',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)

# Optimize hyperparameters
model, params, metrics = trainer.optimize_hyperparameters(
    model_type='xgboost',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    n_trials=100
)

# Save model
trainer.save_model(
    model=model,
    model_name='my_model',
    output_dir='models',
    params=params,
    metrics=metrics,
    feature_names=feature_names
)
```

### ModelEvaluator

Class for comprehensive model evaluation and comparison.

```python
from src.models import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate single model
metrics = evaluator.evaluate_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    model_name='XGBoost'
)

# Compare multiple models
models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model
}

comparison_df = evaluator.compare_models(
    models=models,
    X_test=X_test,
    y_test=y_test
)

# Generate comprehensive report
evaluator.generate_report(
    output_path='models/evaluation',
    models=models,
    X_test=X_test,
    y_test=y_test
)
```

## Feature Engineering Details

### Temporal Features

- `year`: Year of outage
- `month`: Month (1-12)
- `day`: Day of month
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `quarter`: Quarter (1-4)
- `day_of_year`: Day of year (1-365/366)
- `week_of_year`: Week number (1-52)
- `season`: Season (Dry/Wet) based on Kenya climate patterns

### Geographic Features

- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `cluster_id`: Geographic cluster assignment (K-means)

### Target Variable

- `target`: Binary (0/1)
  - 1: Another outage will occur in the same cluster within 7 days
  - 0: No outage in the same cluster within 7 days

## Hyperparameter Optimization

The pipeline uses Optuna for Bayesian optimization of hyperparameters.

### Search Spaces

**Random Forest:**

- n_estimators: [50, 300]
- max_depth: [3, 20]
- min_samples_split: [2, 20]
- min_samples_leaf: [1, 10]
- max_features: ['sqrt', 'log2', None]

**XGBoost:**

- n_estimators: [50, 300]
- learning_rate: [0.01, 0.3]
- max_depth: [3, 10]
- min_child_weight: [1, 10]
- subsample: [0.6, 1.0]
- colsample_bytree: [0.6, 1.0]
- gamma: [0, 5]

**Gradient Boosting:**

- n_estimators: [50, 300]
- learning_rate: [0.01, 0.3]
- max_depth: [3, 10]
- min_samples_split: [2, 20]
- min_samples_leaf: [1, 10]
- subsample: [0.6, 1.0]

## Performance Metrics

### Classification Metrics

- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value (important for reducing false alarms)
- **Recall**: Sensitivity (important for catching actual outages)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (threshold-independent performance)
- **Average Precision**: Area under precision-recall curve

### Model Selection Criteria

The best model is selected based on **ROC-AUC score** on the test set, as it provides a threshold-independent measure of model discrimination ability.

## Troubleshooting

### XGBoost or LightGBM Not Available

If you see warnings about missing libraries:

```bash
# Install XGBoost
pip install xgboost

# Install LightGBM
pip install lightgbm
```

### Optuna Not Available

```bash
pip install optuna
```

### Memory Issues

For large datasets, reduce the number of trials or use a smaller model:

```bash
python scripts/train_model_new.py \
  --model-type logistic_regression \
  --optimize \
  --n-trials 20
```

### Class Imbalance

The models use `class_weight='balanced'` to handle imbalanced classes. You can also adjust the prediction threshold based on your use case:

```python
# Default threshold: 0.5
y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)

# Custom threshold for higher recall
y_pred = (model.predict_proba(X_test)[:, 1] > 0.3).astype(int)
```

## Next Steps

1. **Train your first model**: Run the baseline training script
2. **Analyze results**: Review the evaluation report in `models/evaluation/`
3. **Optimize best model**: Run optimization on the best performing model type
4. **Deploy**: Use the best model for predictions in production
5. **Monitor**: Track model performance over time and retrain as needed

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

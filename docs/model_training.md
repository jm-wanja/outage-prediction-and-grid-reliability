# Model Training

This section describes how machine learning models are trained in the Outage Prediction and Grid Reliability project.

## Overview of the Training Pipeline

1. **Data Loading**: Raw outage data is loaded from `data/kplc_interruption_data.json` using the `load_raw_data` function.
2. **Preprocessing**: Data is cleaned and formatted (coordinates, datetime) with `preprocess_data`.
3. **Feature Engineering**: Temporal, spatial, and historical features are created using `create_temporal_features` and other utilities.
4. **Clustering**: Outages are grouped into clusters (proxies for transformers/zones) using K-means or DBSCAN (`perform_geographic_clustering`).
5. **Target Variable Creation**: The target for prediction is generated (e.g., high outage activity in a zone).
6. **Model Training**: Multiple models are trained and evaluated.
7. **Evaluation**: Performance is measured using metrics like ROC-AUC, precision, recall, and confusion matrix.
8. **Artifacts**: Trained models and evaluation reports are saved in the `models/` directory.

## Supported Models

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **LightGBM**

All models are implemented in `src/models/model_trainer.py` and can be trained via scripts or notebooks.

## How to Run Training

### Command Line

```bash
python scripts/train_model.py --data-path data/kplc_interruption_data.json --output-dir models
```

### Python API

```python
from src.models.train_model import OutagePredictionModel

model = OutagePredictionModel('random_forest')
X_train, X_test, y_train, y_test = model.prepare_data(final_data)
model.train(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
```

## Hyperparameter Tuning

- Grid search and Optuna are supported for hyperparameter optimization.
- Example:

```python
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
tuning_results = model.tune_hyperparameters(X_train, y_train, param_grid)
```

## Evaluation Metrics

- **ROC-AUC**
- **Precision/Recall**
- **Confusion Matrix**
- **Feature Importance**

## Model Artifacts

- Trained models: `models/best_model.joblib`, etc.
- Evaluation reports: `models/evaluation/`

## Example Output

- ROC-AUC Score: 0.87
- Top features: Cluster ID, recent outage count, hour of day
- Confusion matrix and feature importance plots in `models/evaluation/`

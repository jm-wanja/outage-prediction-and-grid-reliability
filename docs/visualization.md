# Visualization

This section describes the visualizations available in the Outage Prediction and Grid Reliability project.

## Types of Visualizations

- **Timeline Analysis**: Daily and monthly outage trends
- **Risk Heat Map**: Geographic visualization of outage probabilities
- **Cluster Map**: Outages grouped by cluster/zone
- **Feature Importance**: Bar plots and SHAP values
- **Performance Metrics**: ROC curves, confusion matrix, precision-recall
- **Interactive Dashboard**: Multi-panel dashboard with all key plots

## How to Generate Visualizations

### From Scripts

- Run EDA and visualization scripts:

```bash
python scripts/run_eda.py --data-path data/kplc_interruption_data.json --interactive
```

- Generate dashboard:

```bash
python scripts/create_dashboard.py
```

### From Python

```python
from src.visualization.visualize import plot_outage_timeline, create_outage_map, create_interactive_dashboard

plot_outage_timeline(df)
outage_map = create_outage_map(df, cluster_col='cluster_id')
outage_map.save('outage_map.html')
dashboard = create_interactive_dashboard(df)
dashboard.write_html('dashboard.html')
```

## Viewing the Dashboard

- The dashboard is published at:
  [View Dashboard](https://jm-wanja.github.io/outage-prediction-and-grid-reliability/dashboard.html)
- Or open `docs/dashboard.html` locally in your browser.

## Example Plots

- Timeline plot: Outages over time
- Cluster map: Outages by region
- Feature importance: Top predictors
- Model performance: ROC-AUC, confusion matrix

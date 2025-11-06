# Outage Prediction and Grid Reliability

Welcome to the documentation for the **Outage Prediction and Grid Reliability** project - a machine learning solution for predicting electrical grid outages in Kenya using historical KPLC data.

## 🎯 Project Goals

This project aims to:

- **Predict outages** before they occur using historical patterns
- **Identify high-risk areas** for targeted maintenance
- **Provide actionable insights** for grid operators
- **Strengthen grid resilience** through data-driven decisions

## 🔍 Key Features

- **Geographic Clustering**: Group outages by location to identify transformer zones
- **Temporal Analysis**: Understand seasonal and daily patterns in outages
- **Predictive Modeling**: Machine learning models for outage probability forecasting
- **Interactive Visualization**: Maps and dashboards for stakeholder insights
- **Reliability Metrics**: Comprehensive grid health indicators

## 📊 Dataset

The project uses the [KPLC Electricity Interruption Data](https://www.kaggle.com/datasets/kingrobi/kplc-electricity-interruption-data-kenya) containing:

- **123,550+ outage records** from across Kenya
- **Geographic coordinates** (latitude, longitude)
- **Timestamp data** for temporal analysis
- **Date range**: 2023-2025 with historical patterns

## 🚀 Quick Start

Get started with just a few commands:

```bash
# Clone the repository
git clone https://github.com/jm-wanja/outage-prediction-grid-reliability.git
cd outage-prediction-grid-reliability

# Set up environment
conda env create -f environment.yml
conda activate outage-prediction

# Install the package
pip install -e .

# Run analysis
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## 🛠️ Technology Stack

- **Python 3.9+** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **XGBoost/LightGBM** - Advanced ML models
- **GeoPandas & Folium** - Geospatial analysis and mapping
- **Streamlit** - Interactive web applications

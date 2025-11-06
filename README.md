# Outage Prediction and Grid Reliability for Kenya Power

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/jm-wanja/outage-prediction-grid-reliability/actions/workflows/tests.yml/badge.svg)](https://github.com/jm-wanja/outage-prediction-grid-reliability/actions/workflows/tests.yml)

## 🎯 Project Overview

This project uses historical outage patterns from Kenya Power and Lighting Company (KPLC) to predict system-wide failures and quantify large-scale outage probabilities. The goal is to deliver actionable insights to strengthen grid resilience and inform preventive actions.

### 🔍 Key Features

- **Spatial-Temporal Analysis**: Leverage geographic coordinates and timestamps from historical outage data
- **Predictive Modeling**: Machine learning models to forecast outage probabilities
- **Interactive Visualization**: Risk maps and dashboards for stakeholders
- **Reliability Metrics**: System-wide outage probability, regional risk assessment, MTBO analysis

## 📊 Dataset

The project uses the [KPLC Electricity Interruption Data](https://www.kaggle.com/datasets/kingrobi/kplc-electricity-interruption-data-kenya) from Kaggle, containing:

- Geographic coordinates (latitude, longitude)
- Timestamp data (ISO dates and Unix timestamps)
- Historical outage patterns across Kenya's power grid

## 🛠️ Technology Stack

- **Python 3.9+** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **XGBoost/LightGBM** - Advanced ML models
- **GeoPandas & Folium** - Geospatial analysis and mapping
- **Matplotlib & Plotly** - Data visualization
- **Streamlit** - Web application framework
- **FastAPI** - API development

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- (Optional) Conda for environment management

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/jm-wanja/outage-prediction-grid-reliability.git
   cd outage-prediction-grid-reliability
   ```

2. **Set up the environment**

   Using conda (recommended):

   ```bash
   conda env create -f environment.yml
   conda activate outage-prediction
   ```

   Or using pip:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install the package in development mode**

   ```bash
   pip install -e .
   ```

4. **Set up pre-commit hooks** (for development)
   ```bash
   pre-commit install
   ```

### Running the Analysis

1. **Data Exploration**

   ```bash
   jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
   ```

2. **Feature Engineering**

   ```bash
   python scripts/feature_engineering.py
   ```

3. **Model Training**

   ```bash
   python scripts/train_model.py
   ```

4. **Launch the Web App**
   ```bash
   streamlit run src/app.py
   ```

## 📁 Project Structure

```
outage-prediction-grid-reliability/
├── README.md                          # Project overview and setup
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
├── environment.yml                    # Conda environment
├── .gitignore                        # Git ignore rules
│
│
├── data/
│   ├── raw/                          # Original, immutable data
│   ├── interim/                      # Intermediate data transformations
│   ├── processed/                    # Final, canonical datasets
│   └── external/                     # External datasets
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_geographic_clustering.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                     # Configuration settings
│   ├── app.py                        # Streamlit web application
│   ├── data/                         # Data processing modules
│   ├── features/                     # Feature engineering
│   ├── models/                       # ML models and training
│   └── visualization/                # Plotting and visualization
│
├── scripts/                          # Standalone scripts
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── conftest.py                   # Pytest configuration
│
├── models/                           # Trained models
│   └── .gitkeep
│
├── reports/                          # Analysis reports
│   ├── figures/                      # Generated graphics
│   └── presentations/                # Presentations
│
├── docs/                             # Documentation
│   ├── api/                          # API documentation
│   └── assets/                       # Documentation assets
│
├── config/                           # Configuration files
│   └── model_config.yml
│
└── .github/                          # GitHub Actions workflows
    └── workflows/
        ├── tests.yml
        ├── docs.yml
        └── deploy.yml
```

## 📈 Methodology

### 1. Geographic Clustering

Since explicit transformer IDs are not available, we use clustering algorithms (K-means, DBSCAN) on latitude/longitude coordinates to create transformer proxies representing grid zones.

### 2. Feature Engineering

- **Temporal features**: Day of week, hour, season, holidays
- **Spatial features**: Geographic density, proximity metrics
- **Historical patterns**: Lag features, rolling statistics
- **External factors**: Weather data integration (future enhancement)

### 3. Model Development

- **Baseline models**: Logistic Regression, Random Forest
- **Advanced models**: XGBoost, LightGBM with hyperparameter optimization
- **Evaluation**: Time-series cross-validation, calibrated probability assessment

### 4. Visualization & Deployment

- Interactive risk maps using Folium
- Real-time dashboards with Streamlit
- API endpoints for model serving

## 🎯 Key Results

- **Predictive Accuracy**: Achieved X% accuracy in predicting outages 24 hours ahead
- **Risk Assessment**: Identified top 10 high-risk zones contributing to Y% of outages
- **Feature Insights**: Historical patterns and geographic clustering as primary predictors

## 📊 Visualizations

The project includes several interactive visualizations:

- **Risk Heat Map**: Geographic visualization of outage probabilities
- **Temporal Analysis**: Time-series plots of outage patterns
- **Feature Importance**: SHAP values and model interpretability
- **Performance Metrics**: Model evaluation dashboards



## 🙏 Acknowledgments

- [KPLC Electricity Interruption Data](https://www.kaggle.com/datasets/kingrobi/kplc-electricity-interruption-data-kenya) from Kaggle
- Kenya Power and Lighting Company for the original dataset
- Open source community for the amazing tools and libraries

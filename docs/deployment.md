# Deployment

This section explains how to deploy and share the results of the Outage Prediction and Grid Reliability project.

## Web Application

- The project includes a Streamlit web app for interactive exploration.
- Launch locally:

```bash
streamlit run src/app.py
```

- The app provides:
  - Interactive risk maps
  - Outage prediction tools
  - Model insights

## API Endpoints

- (If implemented) The project can expose a FastAPI server for programmatic access.
- Launch API server:

```bash
uvicorn src.api:app --reload
```

- Example endpoints:
  - `/predict` - Get outage predictions
  - `/status` - Health check

## GitHub Pages

- Documentation and dashboards are published via GitHub Pages.
- The `docs/` folder is used as the source.
- The dashboard is available at:
  [https://jm-wanja.github.io/outage-prediction-and-grid-reliability/dashboard.html](https://jm-wanja.github.io/outage-prediction-and-grid-reliability/dashboard.html)

## Continuous Integration

- GitHub Actions workflows run tests and build docs on every push.
- See `.github/workflows/` for configuration.

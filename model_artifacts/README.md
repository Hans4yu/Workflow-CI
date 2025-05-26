# Model Artifacts Storage

This directory contains all model training artifacts from automated CI/CD runs.

## Structure
- Each training run creates a timestamped folder (YYYYMMDD_HHMMSS)
- Artifacts include:
  - `model.pkl` - Trained model file
  - `model_info.json` - Model performance metrics
  - `metric_info.json` - Metric descriptions
  - `mlruns/` - MLflow tracking data

## Latest Artifacts
Check the most recent folder for the latest trained model.

## Automated Updates
This directory is automatically updated by GitHub Actions when:
- Manual workflow trigger
- Push to main branch
- Monthly scheduled runs (1st of each month)

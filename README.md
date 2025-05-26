# MLflow CI Workflow for Loan Approval Model

This directory contains an automated CI workflow for model retraining using MLflow Projects. The workflow is triggered through GitHub Actions and can automatically build and deploy a Docker image to Docker Hub.

## ✨ NEW: Permanent Artifacts Storage

**Model artifacts are now automatically saved to the repository!** No more temporary artifacts that disappear after a few days.

### What's Saved

- ✅ **model.pkl** - Trained model file
- ✅ **model_info.json** - Performance metrics (accuracy, precision, recall, etc.)
- ✅ **metric_info.json** - Metric descriptions
- ✅ **mlruns/** - Complete MLflow tracking data
- ✅ **README.md** - Training run details

### Where to Find

All artifacts are stored in the `model_artifacts/` directory with timestamp-based versioning:

```text
model_artifacts/
├── README.md                 # Latest training info
├── 20250526_210000/         # Training run from May 26, 2025 at 21:00:00
│   ├── README.md            # Details for this specific run
│   ├── model.pkl            # The trained model
│   ├── model_info.json      # Performance metrics
│   └── mlruns/              # MLflow tracking data
└── 20250527_140000/         # Next training run
    └── ...
```

## Directory Structure

- **MLProject/** - Contains the MLflow project definition and related files
  - `MLProject` - MLflow project definition file
  - `conda.yaml` - Conda environment specification
  - `modelling.py` - Model training script
  - `loanapproval_preprocessing.csv` - Preprocessed dataset

- **model_artifacts/** - **NEW!** Permanent storage for all training artifacts
- **Dockerfile** - Dockerfile for creating a model serving container
- **docker_hub_link.txt** - Link to the Docker Hub repository

## How It Works

1. **Model Training with MLflow**
   - MLflow Project manages the model training and dependencies
   - Parameters like data path, test size, and CV iterations can be customized
   - Training metrics are tracked in MLflow

2. **Continuous Integration**
   - GitHub Actions workflow automatically runs on:
     - Manual trigger
     - Push to main branch (when relevant files change)
     - Monthly schedule (1st day of each month)
   
3. **Artifact Management**
   - Model artifacts are saved and can be accessed through MLflow
   - Performance metrics, visualizations, and model binaries are stored

## Troubleshooting

### Common Issues and Solutions

1. **MLflow Command Errors**
   - If you encounter errors with the `--no-conda` flag, use the standard `mlflow run` command without this flag. The GitHub Actions workflow has been updated to handle this issue.
   
2. **Path Issues in GitHub Actions**
   - The workflow uses `${{ github.workspace }}/MLProject` as the working directory path. If you're running into path issues, check the debug logs to verify the repository structure.
   
3. **Missing Artifacts**
   - If model artifacts aren't generated correctly, the workflow has fallback mechanisms to ensure the Docker build can still complete.

4. **MLProject File Format**
   - Ensure your MLProject file is properly formatted as YAML. The workflow includes validation to fix common formatting issues automatically.
   
4. **Docker Integration**
   - Trained model is packaged into a Docker container
   - Container includes a Flask API server for model serving
   - Image is pushed to Docker Hub for easy deployment

## Using the Model API

Once deployed, the model exposes these endpoints:

- `GET /` - Basic health check
- `GET /info` - Model metadata and performance metrics
- `POST /predict` - Make predictions with the model

Example prediction request:
```json
{
  "no_of_dependents": -0.5,
  "education": 1,
  "self_employed": 0,
  "income_annum": 0.8,
  "loan_amount": 1.2,
  "loan_term": 0.5,
  "cibil_score": 0.9,
  "residential_assets_value": 0.7,
  "commercial_assets_value": 0.4,
  "luxury_assets_value": 0.3,
  "bank_asset_value": 0.6
}
```

## Running Locally

To run the MLflow project locally:

```bash
cd MLProject
mlflow run . --no-conda -P data_path=loanapproval_preprocessing.csv
```

To build and run the Docker image locally:

```bash
docker build -t loan-approval-model .
docker run -p 5000:5000 loan-approval-model
```

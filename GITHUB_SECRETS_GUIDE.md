# GitHub Actions Secrets Configuration Guide

To use the CI workflow successfully, you need to set up several secrets in your GitHub repository.

## Required Secrets

1. **MLflow Tracking URI and Authentication (if using remote MLflow server)**
   - `MLFLOW_TRACKING_URI`: URL to your MLflow server
   - `MLFLOW_TRACKING_USERNAME`: Username for MLflow authentication
   - `MLFLOW_TRACKING_PASSWORD`: Password for MLflow authentication

2. **Docker Hub Authentication (for pushing Docker images)**
   - `DOCKER_HUB_USERNAME`: Your Docker Hub username
   - `DOCKER_HUB_TOKEN`: Your Docker Hub access token

## How to Add Secrets

1. Go to your GitHub repository
2. Click on **Settings**
3. In the left sidebar, click on **Secrets and variables** → **Actions**
4. Click on **New repository secret**
5. Enter the name and value for each secret
6. Click **Add secret**

## Testing the Workflow

After setting up secrets, you can manually trigger the workflow:

1. Go to the **Actions** tab in your repository
2. Select the **Model Retraining CI** workflow
3. Click **Run workflow**
4. Select the branch and click **Run workflow**

## Workflow Steps

The workflow follows these steps:
1. Set up job
2. Run actions/checkout@v3
3. Set up Python 3.12.7
4. Check Env
5. Install dependencies
6. Run mlflow project
7. Get latest MLflow run_id
8. Install Python dependencies
9. Upload to GitHub
10. Build Docker Model
11. Log in to Docker Hub
12. Tag Docker Image
13. Push Docker Image
14. Post-processing steps
15. Complete job

## Monitoring the Workflow

- Check the **Actions** tab for workflow runs
- Each run will show detailed steps and any errors
- Artifacts are uploaded to GitHub at the end of successful runs
- If Docker push is successful, the image will be available on Docker Hub

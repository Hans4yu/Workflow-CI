import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from datetime import datetime
import joblib

# Set MLflow tracking URI (can be overridden by environment variables in CI)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "Loan Approval CI Workflow"))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and tune loan approval model")
    parser.add_argument("--data_path", type=str, default="loanapproval_preprocessing.csv",
                        help="Path to the preprocessed data CSV")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size for train-test split")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")
    parser.add_argument("--n_iter", type=int, default=20,
                        help="Number of iterations for RandomizedSearchCV")
    return parser.parse_args()

def load_data(data_path, test_size, random_state):
    """Load and prepare the data"""
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Check for any missing values
    print("Missing values in each column:")
    print(data.isnull().sum())
    
    # Remove rows with missing values if any
    data = data.dropna()
    
    # Ensure proper stripping of column names to remove any hidden whitespace
    data.columns = data.columns.str.strip()
    
    # Define features and target
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # Only the last column
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def tune_hyperparameters(X_train, y_train, n_iter, random_state):
    """Tune hyperparameters for the Random Forest model"""
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    model = RandomForestClassifier(random_state=random_state)
    
    # Use RandomizedSearchCV to search the parameter space
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=3,
        verbose=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_, random_search.best_params_, random_search.cv_results_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics"""
    # Start timing for inference latency
    inference_start = time.time()
    y_pred = model.predict(X_test)
    inference_end = time.time()
    inference_latency = inference_end - inference_start
    
    # Prediction probabilities for ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Custom metrics (beyond autolog)
    # 1. Inference latency (ms per sample)
    inference_latency_per_sample = (inference_latency / len(X_test)) * 1000
    
    # 2. Model size
    joblib.dump(model, 'temp_model.joblib')
    model_size_bytes = os.path.getsize('temp_model.joblib')
    model_size_mb = model_size_bytes / (1024 * 1024)
    os.remove('temp_model.joblib')
    
    # 3. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 4. Class distribution in predictions
    pred_class_distribution = pd.Series(y_pred).value_counts().to_dict()
    
    # 5. Feature importance
    feature_importance = model.feature_importances_
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "inference_latency": inference_latency,
        "inference_latency_per_sample": inference_latency_per_sample,
        "model_size_mb": model_size_mb,
        "confusion_matrix": cm,
        "pred_class_distribution": pred_class_distribution,
        "feature_importance": feature_importance
    }

def plot_confusion_matrix(cm, classes):
    """Plot confusion matrix and save it as a figure"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    confusion_matrix_path = os.path.join("artifacts", "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()
    return confusion_matrix_path

def plot_feature_importance(feature_importance, feature_names):
    """Plot feature importance and save it as a figure"""
    # Sort features according to importance
    indices = np.argsort(feature_importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), feature_importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    feature_importance_path = os.path.join("artifacts", "feature_importance.png")
    plt.tight_layout()
    plt.savefig(feature_importance_path)
    plt.close()
    return feature_importance_path

def plot_class_distribution(pred_class_distribution):
    """Plot class distribution and save it as a figure"""
    plt.figure(figsize=(8, 6))
    plt.bar(pred_class_distribution.keys(), pred_class_distribution.values())
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Prediction Class Distribution')
    plt.xticks([0, 1], ['Rejected', 'Approved'])
    
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    class_distribution_path = os.path.join("artifacts", "class_distribution.png")
    plt.tight_layout()
    plt.savefig(class_distribution_path)
    plt.close()
    return class_distribution_path

def save_metric_info():
    """Create a metric info JSON file with descriptions of the metrics"""
    metric_info = {
        "accuracy": "The ratio of correctly predicted instances to the total instances",
        "precision": "The ratio of correctly predicted positive observations to the total predicted positives",
        "recall": "The ratio of correctly predicted positive observations to all observations in the actual class",
        "f1_score": "The weighted average of precision and recall",
        "roc_auc": "Area under the ROC Curve, representing model's ability to discriminate between classes",
        "inference_latency": "Total time taken to make predictions on the test set in seconds",
        "inference_latency_per_sample": "Average time taken to make a prediction for one sample in milliseconds",
        "model_size_mb": "Size of the trained model in megabytes",
        "feature_importance": "Relative importance of each feature in the model's decision making"
    }
    
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    metric_info_path = os.path.join("artifacts", "metric_info.json")
    
    with open(metric_info_path, "w") as f:
        json.dump(metric_info, f, indent=4)
    
    return metric_info_path

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(
        args.data_path, 
        args.test_size, 
        args.random_state
    )
    
    print("Tuning hyperparameters...")
    with mlflow.start_run(run_name=f"RandomForest_CI_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_iter", args.n_iter)
        
        # Tune model
        tuned_model, best_params, cv_results = tune_hyperparameters(
            X_train, 
            y_train, 
            args.n_iter, 
            args.random_state
        )
        
        # Log best parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # Evaluate model
        metrics = evaluate_model(tuned_model, X_test, y_test)
        
        # Plot confusion matrix
        classes = [0, 1]
        confusion_matrix_path = plot_confusion_matrix(metrics["confusion_matrix"], classes)
        
        # Plot feature importance
        feature_names = X_train.columns
        feature_importance_path = plot_feature_importance(metrics["feature_importance"], feature_names)
        
        # Plot class distribution
        class_distribution_path = plot_class_distribution(metrics["pred_class_distribution"])
        
        # Create metric info JSON
        metric_info_path = save_metric_info()
        
        # Log standard metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        
        # Log additional custom metrics
        mlflow.log_metric("inference_latency", metrics["inference_latency"])
        mlflow.log_metric("inference_latency_per_sample", metrics["inference_latency_per_sample"])
        mlflow.log_metric("model_size_mb", metrics["model_size_mb"])
        
        # Log feature importance values individually
        for i, importance in enumerate(metrics["feature_importance"]):
            mlflow.log_metric(f"feature_importance_{feature_names[i]}", importance)
        
        # Log cv results as artifact
        cv_results_df = pd.DataFrame(cv_results)
        os.makedirs("artifacts", exist_ok=True)
        cv_results_path = os.path.join("artifacts", "cv_results.csv")
        cv_results_df.to_csv(cv_results_path, index=False)
        
        # Log artifacts
        mlflow.log_artifact(confusion_matrix_path)
        mlflow.log_artifact(feature_importance_path)
        mlflow.log_artifact(class_distribution_path)
        mlflow.log_artifact(metric_info_path)
        mlflow.log_artifact(cv_results_path)
        
        # Save model in pickle format
        model_path = os.path.join("artifacts", "model.pkl")
        joblib.dump(tuned_model, model_path)
        mlflow.log_artifact(model_path)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(
            tuned_model, 
            "model"
        )
        
        print(f"Model trained with accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Inference Latency: {metrics['inference_latency']:.6f} seconds")
        print(f"Inference Latency per sample: {metrics['inference_latency_per_sample']:.6f} ms")
        print(f"Model Size: {metrics['model_size_mb']:.6f} MB")
        print(f"Best parameters: {best_params}")
        
        # Save model info for easy reference
        model_info = {
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score'],
            "roc_auc": metrics['roc_auc'],
            "best_params": best_params,
            "model_size_mb": metrics['model_size_mb'],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        model_info_path = os.path.join("artifacts", "model_info.json")
        with open(model_info_path, "w") as f:
            json.dump(model_info, f, indent=4)
        mlflow.log_artifact(model_info_path)

if __name__ == "__main__":
    main()

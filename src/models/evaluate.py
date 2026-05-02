import mlflow
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints relevant metrics.
    
    Args:
        model: The trained model to evaluate.
        X_test: Test features.
        y_test: True labels for the test set.
    """
    print("Evaluating model performance on the test set...")
    
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["eval"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["eval"]["experiment_name"])

    with mlflow.start_run(run_name="Model Evaluation"):
        
        
        # Predict probabilities and classes
        print("Generating predictions...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        print("Calculating evaluation metrics...")
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("roc_auc", roc_auc)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        mlflow.log_metric("balanced_accuracy", balanced_acc)
        
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Balanced Accuracy Score: {balanced_acc:.4f}")
        print("Classification Report:")
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        mlflow.log_text(str(confusion_matrix(y_test, y_pred)), "confusion_matrix.txt")
        print(confusion_matrix(y_test, y_pred))

        print("Model evaluation completed and logged to MLflow.")
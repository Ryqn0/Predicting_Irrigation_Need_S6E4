import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

from src.features.build_final import build_final_features


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config: dict, train_df: pd.DataFrame):

    print("Starting training with the following configuration:")
    print(yaml.dump(config))

    print("Building features for training and test data...")
    train_df = train_df.copy()

    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=config["training"]["random_state"], stratify=train_df["Irrigation_Need"])

    train_df, train_enc = build_final_features(train_df)
    test_df = test_df.copy()
    test_df, _ = build_final_features(test_df, encodings=train_enc)
    print("Finished building features.")

    y_train = train_df["Irrigation_Need"]
    X_train = train_df.drop(columns=["Irrigation_Need"])
    y_test = test_df["Irrigation_Need"]
    X_test = test_df.drop(columns=["Irrigation_Need"])  


    print("Training model...")
    model = lgb.LGBMClassifier(**config["model"]["params"])

    print("Setting up MLflow tracking...")
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        print("Logging model parameters...")
        mlflow.log_params(config["model"]["params"])
        

        scores = cross_val_score(
            model, X_train, y_train,
            cv=config["training"]["cv_folds"],
            scoring="balanced_accuracy_score",
        )
        print(f"Cross-validation balanced accuracy scores: {scores}")
        mlflow.log_metric("cv_balanced_accuracy_mean", scores.mean())
        mlflow.log_metric("cv_balanced_accuracy_std", scores.std())
        print("Finished cross-validation.")
        print("Fitting model on training data...")

        model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=200)
        ]
        )
        print("Finished training")

        print("Evaluating model on test data...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"Test Balanced Accuracy: {balanced_acc:.4f}")
        mlflow.log_metric("test_balanced_accuracy", balanced_acc)
        print(f"Test ROC AUC: {roc_auc:.4f}")
        mlflow.log_metric("test_roc_auc", roc_auc)
        print("Classification Report:")
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        mlflow.log_text(str(confusion_matrix(y_test, y_pred)), "confusion_matrix.txt")
        print(confusion_matrix(y_test, y_pred))

        mlflow.sklearn.log_model(model, "model")

        # Save the model locally as well
        model_path = Path(config["model"]["save_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        return model
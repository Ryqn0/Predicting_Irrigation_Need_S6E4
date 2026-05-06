import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

from src.features.build_final import build_final_features


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def train(CONFIG_PATH: str, train_df: pd.DataFrame):

    config = load_config(CONFIG_PATH)

    print("Starting training with the following configuration:")
    print(yaml.dump(config))

    print("Building features for training and test data...")
    train_df = train_df.copy()

    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=config["training"]["random_state"], stratify=train_df["Irrigation_Need"])

    train_df, train_enc = build_final_features(train_df)
    test_df = test_df.copy()
    test_df, _ = build_final_features(test_df, encodings=train_enc)
    print("Finished building features.")

    drop_cols = [c for c in ["Irrigation_Need", "id"] if c in train_df.columns]
    y_train = train_df["Irrigation_Need"]
    X_train = train_df.drop(columns=drop_cols)
    y_test = test_df["Irrigation_Need"]
    X_test = test_df.drop(columns=drop_cols)

    params = config["model"]["params"]
    n_estimators = params.pop("n_estimators")

    print("Training model...")
    print("Setting up MLflow tracking...")
    mlflow.set_tracking_uri(config["mlflow"]["train"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["train"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({**params, "n_estimators": n_estimators})

        # CV with early stopping — finds best round count without training 10k trees per fold
        print("Running lgb.cv with early stopping...")
        cv_result = lgb.cv(
            params,
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=n_estimators,
            nfold=config["training"]["cv_folds"],
            stratified=True,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=200),
            ],
            seed=config["training"]["random_state"],
        )
        best_n = len(next(iter(cv_result.values())))
        cv_mean = next(v for k, v in cv_result.items() if k.endswith("-mean"))[-1]
        cv_std  = next(v for k, v in cv_result.items() if k.endswith("-stdv"))[-1]
        print(f"Best round: {best_n}  |  CV logloss: {cv_mean:.4f} ± {cv_std:.4f}")
        mlflow.log_metrics({"cv_logloss_mean": cv_mean, "cv_logloss_std": cv_std, "best_n_estimators": best_n})

        # Final fit on full train set using the best round count
        print("Fitting final model...")
        model = lgb.LGBMClassifier(**params, n_estimators=best_n)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(period=200)],
        )
        print("Finished training")

        print("Evaluating model on test data...")
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)          # shape (n, 3)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
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
        mlflow.log_artifact(str(model_path), artifact_path="model")
        print(f"Model saved to {model_path}")

        # Save features columns for later use in prediction
        features_path = model_path.with_suffix(".features")
        features_txt_path = model_path.with_suffix(".txt")
        joblib.dump(X_train.columns, features_path)
        features_txt_path.write_text("\n".join(X_train.columns), encoding="utf-8")
        mlflow.log_artifact(str(features_txt_path), artifact_path="features")
        mlflow.log_artifact(str(features_path), artifact_path="features")
        print(f"Features saved to {features_path}")

        # Save encodings for later use in prediction
        encodings_path = model_path.with_suffix(".encodings")
        joblib.dump(train_enc, encodings_path)
        mlflow.log_artifact(str(encodings_path), artifact_path="encodings")
        print(f"Encodings saved to {encodings_path}")

        return model
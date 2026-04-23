import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.data.load import load_processed
from src.features.build import build_features
from src.evaluation.metrics import summarize


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    df = load_processed("train.csv")
    df = build_features(df)

    target = "target"  # TODO: update with actual target column name
    X = df.drop(columns=[target])
    y = df[target]

    model = XGBClassifier(**config["model"]["params"])

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(config["model"]["params"])

        scores = cross_val_score(
            model, X, y,
            cv=config["training"]["cv_folds"],
            scoring="roc_auc",
        )
        mlflow.log_metric("cv_roc_auc_mean", scores.mean())
        mlflow.log_metric("cv_roc_auc_std", scores.std())

        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model")

        print(summarize(scores))


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)

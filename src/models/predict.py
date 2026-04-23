import pandas as pd
import mlflow.sklearn

from src.data.load import load_processed
from src.features.build import build_features


def predict(run_id: str, output_path: str = "reports/submission.csv") -> None:
    df = load_processed("test.csv")
    df = build_features(df)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    preds = model.predict_proba(df)[:, 1]

    submission = pd.DataFrame({"id": df.index, "target": preds})
    submission.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

import pandas as pd
from pathlib import Path


def load_raw(filename: str) -> pd.DataFrame:
    path = Path("data/raw") / filename
    return pd.read_csv(path)


def load_processed(filename: str) -> pd.DataFrame:
    path = Path("data/processed") / filename
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    path = Path("data/processed") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

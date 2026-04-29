import pandas as pd
from pathlib import Path


def load_data(path_to_file: str) -> pd.DataFrame:
    print(f"Loading data from {path_to_file}...")
    if not Path(path_to_file).exists():
        raise FileNotFoundError(f"File {path_to_file} does not exist.")
    else:
        print(f"File {path_to_file} found.")
    df = pd.read_csv(path_to_file)
    print(f"Data loaded successfully with {len(df)} records and {len(df.columns)} columns.")
    return df
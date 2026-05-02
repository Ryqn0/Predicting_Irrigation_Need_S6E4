import pandas as pd

from src.data.load import load_data
from src.data.verify import verify_data
from src.features.build_final import build_final_features


def test_complete_preprocess(path_to_file="../data/raw/train.csv") -> pd.DataFrame:
    df = load_data(path_to_file)
    df = verify_data(df)
    df, _ = build_final_features(df)

    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"

    return df

if __name__ == "__main__":
    df = test_complete_preprocess()
    print("Preprocessing test completed successfully.")
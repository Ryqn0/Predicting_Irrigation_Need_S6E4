import sys

sys.path.append("src")

from src.data.load import load_data
from src.data.verify import verify_data
from src.features.build_final import build_final_features

def main():
    # Load and verify data
    df = load_data("../data/raw/train.csv")
    df = verify_data(df)

    # Build features
    df, _ = build_final_features(df)

    # Save processed data
    df.to_csv("../data/processed/train_processed.csv", index=False)
    print("Training data preprocessing completed and saved to ../data/processed/train_processed.csv")
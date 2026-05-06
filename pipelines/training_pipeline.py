import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split

from src.data.load import load_data
from src.data.verify import verify_data
from src.models.train import load_config, train
from src.models.evaluate import evaluate_model

TRAIN_DATA_PATH = "../data/raw/train.csv"
DATA_PATH = "../data/processed/train_processed.csv"
CONFIG_PATH = "../configs/config.yaml"

def main():
    # Load and verify data
    df = load_data(TRAIN_DATA_PATH)
    df = verify_data(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Irrigation_Need"])

    # Load training configuration
    config = load_config(CONFIG_PATH)

    # Train model
    model = train(CONFIG_PATH, train_df=train_df)

    encodings_path = Path(config["model"]["save_path"]).with_suffix(".encodings")
    encodings = None
    if encodings_path.exists():
        encodings = joblib.load(encodings_path)
        print(f"Loaded encodings from {encodings_path}")
    else:
        print(f"No encodings file found at {encodings_path}. Evaluation will proceed without encodings.")

    # Evaluate model
    evaluate_model(model, test_df, encodings=encodings)

    # Save processed data
    df.to_csv(DATA_PATH, index=False)
    print("Training data preprocessing completed and saved to {}".format(DATA_PATH))

if __name__ == "__main__":
    main()
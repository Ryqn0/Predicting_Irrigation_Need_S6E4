import pandas as pd
import joblib

from src.models.predict import predict

MODEL_PATH = "../pipelines/models/lightgbm_model.pkl"
MODEL_ENCODINGS_PATH = "../pipelines/models/lightgbm_model.encodings"
MODEL_FEATURES_PATH = "../pipelines/models/lightgbm_model.features"

def test_predict():
    model = joblib.load(MODEL_PATH)
    model_encodings = joblib.load(MODEL_ENCODINGS_PATH)
    model_features = joblib.load(MODEL_FEATURES_PATH)

    input_data = pd.read_csv("../data/raw/test.csv").iloc[2].to_dict()

    result = predict(model, model_encodings, model_features, input_data)
    print(result)
    assert isinstance(result, str)
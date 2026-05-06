import pandas as pd

from src.data.verify import verify_data
from src.features.build_final import build_final_features


def predict(model, model_encodings, input = dict) :

    input = pd.DataFrame(input, index=[0])
    input = verify_data(input)
    input = input.drop(columns=["id"], errors="ignore")
    input, _ = build_final_features(input, encodings=model_encodings)

    preds = model.predict(input)
    preds_proba = model.predict_proba(input)

    if preds == 0:
        return f"Low irrigation need with probability {preds_proba[0][0]:.2f}"

    elif preds == 1:
        return f"Medium irrigation need with probability {preds_proba[0][1]:.2f}"
    
    elif preds == 2:
        return f"High irrigation need with probability {preds_proba[0][2]:.2f}"
    
    else:
        return f"Unknown prediction: {preds[0]}"

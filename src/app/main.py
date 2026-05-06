import pydantic
from fastapi import FastAPI
import joblib
import gradio as gr
from pathlib import Path

from src.models.predict import predict

PIPELINE_MODEL_PATH = Path(__file__).resolve().parents[2] / "pipelines" / "models"

model = joblib.load(PIPELINE_MODEL_PATH / "lightgbm_model.pkl")
model_encodings = joblib.load(PIPELINE_MODEL_PATH / "lightgbm_model.encodings")
model_features = joblib.load(PIPELINE_MODEL_PATH / "lightgbm_model.features")

app = FastAPI(
    title="Irrigation Need Prediction API",
    description="An API for predicting irrigation needs using a pre-trained model.",
    version="1.0.0"
)

# Data schema for prediction
class AgriculturalEnvironment(pydantic.BaseModel):
    id: int | None = None
    Soil_Type: str
    Soil_pH: float
    Soil_Moisture: float
    Organic_Carbon: float
    Electrical_Conductivity: float
    Temperature_C: float
    Humidity: float
    Rainfall_mm: float
    Sunlight_Hours: float
    Wind_Speed_kmh: float
    Crop_Type: str
    Crop_Growth_Stage: str
    Season: str
    Irrigation_Type: str
    Water_Source: str
    Field_Area_hectare: float
    Mulching_Used: str
    Previous_Irrigation_mm: float
    Region: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/model_info")
def model_info():
    return {
        "model_name": "LightGBM",
        "model_version": "1.0.0",
        "is_model_loaded": model is not None
    }

@app.post("/predict")
def predict_endpoint(input_data: AgriculturalEnvironment):
    input_dict = input_data.model_dump()  # Convert Pydantic model to dict
    prediction = predict(model, model_encodings, model_features, input=input_dict)
    return {"prediction": prediction}

def predict_fn(Soil_Type: str, Soil_pH: float, Soil_Moisture: float, Organic_Carbon: float,
               Electrical_Conductivity: float, Temperature_C: float, Humidity: float, Rainfall_mm: float,
               Sunlight_Hours: float, Wind_Speed_kmh: float, Crop_Type: str, Crop_Growth_Stage: str,
               Season: str, Irrigation_Type: str, Water_Source: str, Field_Area_hectare: float,
               Mulching_Used: str, Previous_Irrigation_mm: float, Region: str) -> str:
    data = {
        "Soil_Type": Soil_Type,
        "Soil_pH": Soil_pH,
        "Soil_Moisture": Soil_Moisture,
        "Organic_Carbon": Organic_Carbon,
        "Electrical_Conductivity": Electrical_Conductivity,
        "Temperature_C": Temperature_C,
        "Humidity": Humidity,
        "Rainfall_mm": Rainfall_mm,
        "Sunlight_Hours": Sunlight_Hours,
        "Wind_Speed_kmh": Wind_Speed_kmh,
        "Crop_Type": Crop_Type,
        "Crop_Growth_Stage": Crop_Growth_Stage,
        "Season": Season,
        "Irrigation_Type": Irrigation_Type,
        "Water_Source": Water_Source,
        "Field_Area_hectare": Field_Area_hectare,
        "Mulching_Used": Mulching_Used,
        "Previous_Irrigation_mm": Previous_Irrigation_mm,
        "Region": Region
    }
    return predict(model, model_encodings, model_features, input=data)

# GRADIO UI COMPONENTS
gradio_app = gr.Interface(
    fn=predict_fn,
    inputs=[
        gr.Dropdown(choices=['Sandy', 'Clay', 'Silt', 'Loamy'], label="Soil Type"),
        gr.Slider(minimum=3.5, maximum=9.0, step=0.01, label="Soil pH"),
        gr.Slider(minimum=0, maximum=100, step=0.01, label="Soil Moisture (%)"),
        gr.Slider(minimum=0, maximum=2, step=0.01, label="Organic Carbon (%)"),
        gr.Slider(minimum=0, maximum=4, step=0.01, label="Electrical Conductivity (dS/m)"),
        gr.Slider(minimum=0, maximum=50, step=0.01, label="Temperature (°C)"),
        gr.Slider(minimum=0, maximum=100, step=0.01, label="Humidity (%)"),
        gr.Slider(minimum=0, maximum=2600, step=0.01, label="Rainfall (mm)"),
        gr.Slider(minimum=3, maximum=16, step=0.01, label="Sunlight Hours"),
        gr.Slider(minimum=0, maximum=25, step=0.01, label="Wind Speed (km/h)"),
        gr.Dropdown(choices=['Maize', 'Cotton', 'Wheat', 'Rice', 'Sugarcane', 'Potato'], label="Crop Type"),
        gr.Dropdown(choices=['Sowing', 'Vegetative', 'Flowering', 'Harvest'], label="Crop Growth Stage"),
        gr.Dropdown(choices=['Rabi', 'Kharif', 'Zaid'], label="Season"),
        gr.Dropdown(choices=['Canal', 'Drip', 'Sprinkler', 'Rainfed'], label="Irrigation Type"),
        gr.Dropdown(choices=['River', 'Reservoir', 'Groundwater', 'Rainwater'], label="Water Source"),
        gr.Slider(minimum=0, maximum=20, step=0.01, label="Field Area (hectares)"),
        gr.Dropdown(choices=["Yes", "No"], label="Mulching Used"),
        gr.Slider(minimum=0, maximum=130, step=0.01, label="Previous Irrigation (mm)"),
        gr.Dropdown(choices=['North', 'South', 'East', 'West', 'Central'], label="Region"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Irrigation Need Prediction",
    description="Enter agricultural data to predict the irrigation need level (Low, Medium, High).",
    examples=[
        ["Clay", 4.93, 20.28, 1.06, 2.93, 14.64, 81.96, 2175.11, 5.88, 11.32, "Rice", "Flowering", "Rabi", "Rainfed", "Rainwater", 5.11, "No", 4.91, "West"],
        ["Sandy", 6.5, 30.0, 0.5, 1.0, 25.0, 60.0, 500.0, 8.0, 5.0, "Wheat", "Vegetative", "Kharif", "Drip", "Groundwater", 10.0, "Yes", 20.0, "North"],
        ["Loamy", 5.5, 15.0, 1.5, 3.0, 30.0, 70.0, 1000.0, 10.0, 15.0, "Maize", "Sowing", "Zaid", "Sprinkler", "River", 2.5, "No", 10.0, "South"]

    ]
)

# MOUNT THE GRADIO INTERFACE TO THE FASTAPI APP
app = gr.mount_gradio_app(app, gradio_app, path="/ui")



from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["model_name"] == "LightGBM"
    assert json_response["model_version"] == "1.0.0"
    assert json_response["is_model_loaded"] is True

def test_predict():
    input_data = {
        #"id": None,
        "Soil_Type": "Clay",
        "Soil_pH": 4.93,
        "Soil_Moisture": 20.28,
        "Organic_Carbon": 1.06,
        "Electrical_Conductivity": 2.93,
        "Temperature_C": 14.64,
        "Humidity": 81.96,
        "Rainfall_mm": 2175.11,
        "Sunlight_Hours": 5.88,
        "Wind_Speed_kmh": 11.32,
        "Crop_Type": "Rice",
        "Crop_Growth_Stage": "Flowering",
        "Season": "Rabi",
        "Irrigation_Type": "Rainfed",
        "Water_Source": "Rainwater",
        "Field_Area_hectare": 5.11,
        "Mulching_Used": "No",
        "Previous_Irrigation_mm": 4.91,
        "Region": "West"
    }
    response = client.post("/predict", json=input_data)
    print(response.json())
    assert response.status_code == 200
    assert "prediction" in response.json()

if __name__ == "__main__":
    test_health_check()
    test_model_info()
    test_predict()
    print("All tests passed!")


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Resolve the model path relative to this file so the API works regardless of cwd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "namma_yatri_model.pkl")

app = FastAPI(
    title="Namma Yatri Dynamic Pricing API",
    description="Real-time inference API for predicting neighborhood ride-hailing fares.",
    version="1.0.0"
)

# Load the model once at startup rather than per-request
print(f"Loading model from {MODEL_PATH}...")
try:
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline['model']
    scaler = pipeline['scaler']
    print("Model loaded successfully.")
except FileNotFoundError:
    raise RuntimeError("Model file not found. Run train_mlflow.py first to generate it.")

class RideRequest(BaseModel):
    distance_km: float
    completed_trips: float
    cancellation_rate: float 

@app.get("/")
def home():
    return {"message": "Namma Yatri Pricing API is live! Send a POST request to /predict."}

@app.post("/predict")
def predict_fare(request: RideRequest):
    try:
        input_data = np.array([[
            request.distance_km, 
            request.completed_trips, 
            request.cancellation_rate
        ]])
        
        # Scale using the same scaler fit during training
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        
        return {
            "predicted_fare_inr": round(prediction, 2),
            "currency": "INR",
            "model_version": "GradientBoosting_FastLearner_v1"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

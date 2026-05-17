from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# 1. Setup robust file paths so the API finds your model from any folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "namma_yatri_model.pkl")

# 2. Initialize the API
app = FastAPI(
    title="Namma Yatri Dynamic Pricing API",
    description="Real-time inference API for predicting neighborhood ride-hailing fares.",
    version="1.0.0"
)

# 3. Load the model and scaler into memory right when the server boots up
print(f"Loading model from {MODEL_PATH}...")
try:
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline['model']
    scaler = pipeline['scaler']
    print("✅ Champion Model loaded successfully!")
except FileNotFoundError:
    raise RuntimeError("Model file not found. Ensure your train script saved it correctly!")

# 4. Define the exact JSON structure we expect from frontends/users
class RideRequest(BaseModel):
    distance_km: float
    completed_trips: float
    cancellation_rate: float 

# Health check endpoint
@app.get("/")
def home():
    return {"message": "Namma Yatri Pricing API is live! Send a POST request to /predict."}

# The main inference endpoint
@app.post("/predict")
def predict_fare(request: RideRequest):
    try:
        # Extract data from the incoming request
        input_data = np.array([[
            request.distance_km, 
            request.completed_trips, 
            request.cancellation_rate
        ]])
        
        # Scale the data using the exact scaler from our training phase
        scaled_data = scaler.transform(input_data)
        
        # Make the prediction
        prediction = model.predict(scaled_data)[0]
        
        # Return a clean JSON response
        return {
            "predicted_fare_inr": round(prediction, 2),
            "currency": "INR",
            "model_version": "GradientBoosting_FastLearner_v1"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
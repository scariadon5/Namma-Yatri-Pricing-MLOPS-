import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
DATA_PATH = "../data/raw/namma_yatri.csv"
MODEL_DIR = "../models/"

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['Completed Trips'] = df['Completed Trips'].astype(str).str.replace(',', '').astype(float)
    df['Booking Cancellation Rate'] = df['Booking Cancellation Rate'].astype(str).str.replace('%', '').astype(float) / 100
    df['Average Fare per Trip'] = df['Average Fare per Trip'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    features = ['Average Distance per Trip (km)', 'Completed Trips', 'Booking Cancellation Rate']
    target = 'Average Fare per Trip'
    return df[features + [target]].dropna()

def train_with_mlflow():
    # 1. Set up MLflow to save data locally to a SQLite file
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Namma_Yatri_Dynamic_Pricing")
    
    df = load_and_clean_data(DATA_PATH)
    
    X = df.drop('Average Fare per Trip', axis=1)
    y = df['Average Fare per Trip']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # --- MLFLOW TRACKING STARTS HERE ---
    with mlflow.start_run():
        print("Training Model with MLflow Tracking...")
        
        # Hyperparameters (These are what you tweak!)
        n_estimators = 50
        learning_rate = 0.2
        max_depth = 4
        
        # Log parameters to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        
        # Train Model
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        print(f"RMSE: ₹{rmse:.2f} | R2: {r2:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log the actual model file into MLflow's registry
        mlflow.sklearn.log_model(model, "model")
        
        # Save a local fallback copy for our API later
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({'model': model, 'scaler': scaler}, os.path.join(MODEL_DIR, "namma_yatri_model.pkl"))
        
        print("✅ Run successfully tracked in MLflow!")

if __name__ == "__main__":
    train_with_mlflow()
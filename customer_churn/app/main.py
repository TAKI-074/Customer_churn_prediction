from fastapi import FastAPI
import pickle
import numpy as np
from pathlib import Path

app = FastAPI(title="Customer Churn Prediction API")

# Root endpoint
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}

print("📦 Loading model files...")

# Correct model path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

print("Model directory:", MODEL_DIR)

model = pickle.load(open(MODEL_DIR / "churn_model.pkl", "rb"))
scaler = pickle.load(open(MODEL_DIR / "scaler.pkl", "rb"))
FEATURE_ORDER = pickle.load(open(MODEL_DIR / "feature_order.pkl", "rb"))

print("✅ Model files loaded")

@app.post("/predict")
def predict(data: dict):
    try:
        input_data = [data[feature] for feature in FEATURE_ORDER]
    except KeyError as e:
        return {"error": f"Missing feature: {str(e)}"}

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3)
    }

@app.get("/test")
def test():
    return {"ok": True}

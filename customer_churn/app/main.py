'''from fastapi import FastAPI
import pickle
import numpy as np

# Create FastAPI app
app = FastAPI(title="Customer Churn Prediction API")

# Load trained artifacts
model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
feature_order = pickle.load(open("model/feature_order.pkl", "rb"))

# Business decision threshold
THRESHOLD = 0.35

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    """
    Expects JSON with keys exactly matching training feature names
    """

    try:
        # Maintain same feature order as training
        input_data = [data[feature] for feature in feature_order]
    except KeyError as e:
        return {"error": f"Missing feature: {str(e)}"}

    # Convert to numpy array
    input_array = np.array(input_data).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict
    churn_probability = model.predict_proba(input_scaled)[0][1]
    churn_prediction = int(churn_probability >= THRESHOLD)

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": round(float(churn_probability), 3)
    }'''

'''
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Customer Churn Prediction API")

# Load model files
with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/feature_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

# Input schema
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    input_data = np.array([[getattr(data, feature) for feature in feature_order]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3)
    }
'''

print("🚀 main.py started")

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

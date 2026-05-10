from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path

app = FastAPI(title="Customer Churn Prediction API")

# -----------------------------
# Load Model Files
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

model = pickle.load(open(MODEL_DIR / "churn_model.pkl", "rb"))
scaler = pickle.load(open(MODEL_DIR / "scaler.pkl", "rb"))
FEATURE_ORDER = pickle.load(open(MODEL_DIR / "feature_order.pkl", "rb"))

# -----------------------------
# Root Endpoint
# -----------------------------

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running"
    }

# -----------------------------
# Health Check Endpoint
# -----------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

# -----------------------------
# Input Validation Model
# -----------------------------

class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

    gender_Male: int

    Partner_Yes: int
    Dependents_Yes: int

    PhoneService_Yes: int

    MultipleLines_No_phone_service: int
    MultipleLines_Yes: int

    InternetService_Fiber_optic: int
    InternetService_No: int

    OnlineSecurity_No_internet_service: int
    OnlineSecurity_Yes: int

    OnlineBackup_No_internet_service: int
    OnlineBackup_Yes: int

    DeviceProtection_No_internet_service: int
    DeviceProtection_Yes: int

    TechSupport_No_internet_service: int
    TechSupport_Yes: int

    StreamingTV_No_internet_service: int
    StreamingTV_Yes: int

    StreamingMovies_No_internet_service: int
    StreamingMovies_Yes: int

    Contract_One_year: int
    Contract_Two_year: int

    PaperlessBilling_Yes: int

    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    try:
        input_data = [
            customer.model_dump()[feature]
            for feature in FEATURE_ORDER
        ]

    except KeyError as e:
        return {
            "error": f"Missing feature: {str(e)}"
        }

    input_array = np.array(input_data).reshape(1, -1)

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3)
    }
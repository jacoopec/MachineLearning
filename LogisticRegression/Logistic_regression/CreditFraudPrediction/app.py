# app.py

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Load model and scaler (once)
# ----------------------------

MODEL_PATH = "logistic_credit_model.joblib"
SCALER_PATH = "credit_scaler.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "income",
    "debt_to_income_ratio",
    "credit_utilization",
    "late_payments",
    "loan_amount",
    "employment_length",
    "age"
]

# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="Credit Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request / Response schemas
# ----------------------------

class CreditRequest(BaseModel):
    income: float = Field(..., example=60000)
    debt_to_income_ratio: float = Field(..., example=0.35)
    credit_utilization: float = Field(..., example=0.55)
    late_payments: int = Field(..., example=1)
    loan_amount: float = Field(..., example=12000)
    employment_length: int = Field(..., example=6)
    age: int = Field(..., example=38)


class CreditResponse(BaseModel):
    probability_default: float
    predicted_label: int
    decision: str


# ----------------------------
# Prediction logic
# ----------------------------

def predict_default(data: CreditRequest):
    df = pd.DataFrame([[getattr(data, f) for f in FEATURES]], columns=FEATURES)

    X_scaled = scaler.transform(df)
    prob_default = model.predict_proba(X_scaled)[0, 1]
    label = int(prob_default >= 0.5)

    return prob_default, label


# ----------------------------
# API endpoint
# ----------------------------

@app.post("/predict", response_model=CreditResponse)
def predict(data: CreditRequest):
    try:
        prob, label = predict_default(data)
        return CreditResponse(
            probability_default=round(prob, 4),
            predicted_label=label,
            decision="default" if label == 1 else "no default"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

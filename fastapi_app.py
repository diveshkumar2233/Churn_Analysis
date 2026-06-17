"""
FastAPI Backend — Customer Churn Prediction
Place your .pkl files in the /models/ folder alongside this file.
Run: uvicorn fastapi_app:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle, numpy as np, os

app = FastAPI(title="Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Load PKL files ────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

def load(name):
    p = os.path.join(BASE, "models", name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

model         = load("customer_churn_model.pkl")
label_encoder = load("label_encoder.pkl")
feature_cols  = load("feature_columns.pkl")

# ── Input schema ──────────────────────────────────────────────────────────────
class Customer(BaseModel):
    gender:            str   = Field(..., example="Male")
    senior_citizen:    int   = Field(..., ge=0, le=1, example=0)
    partner:           str   = Field(..., example="Yes")
    dependents:        str   = Field(..., example="No")
    tenure:            int   = Field(..., ge=0, example=12)
    phone_service:     str   = Field(..., example="Yes")
    multiple_lines:    str   = Field(..., example="No")
    internet_service:  str   = Field(..., example="Fiber optic")
    online_security:   str   = Field(..., example="No")
    online_backup:     str   = Field(..., example="Yes")
    device_protection: str   = Field(..., example="No")
    tech_support:      str   = Field(..., example="No")
    streaming_tv:      str   = Field(..., example="No")
    streaming_movies:  str   = Field(..., example="No")
    contract:          str   = Field(..., example="Month-to-month")
    paperless_billing: str   = Field(..., example="Yes")
    payment_method:    str   = Field(..., example="Electronic check")
    monthly_charges:   float = Field(..., ge=0, example=70.35)
    total_charges:     float = Field(..., ge=0, example=845.50)

# ── Encoding maps ─────────────────────────────────────────────────────────────
YN       = {"Yes": 1, "No": 0}
GEN      = {"Male": 1, "Female": 0}
INET     = {"DSL": 0, "Fiber optic": 1, "No": 2}
CONTRACT = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT  = {"Electronic check": 0, "Mailed check": 1,
             "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}
ML       = {"No phone service": 0, "No": 1, "Yes": 2}
ISVC     = {"No internet service": 0, "No": 1, "Yes": 2}

def encode(c: Customer) -> np.ndarray:
    row = [
        GEN.get(c.gender, 0), c.senior_citizen,
        YN.get(c.partner, 0), YN.get(c.dependents, 0),
        c.tenure,
        YN.get(c.phone_service, 0),
        ML.get(c.multiple_lines, 1),
        INET.get(c.internet_service, 0),
        ISVC.get(c.online_security, 1),
        ISVC.get(c.online_backup, 1),
        ISVC.get(c.device_protection, 1),
        ISVC.get(c.tech_support, 1),
        ISVC.get(c.streaming_tv, 1),
        ISVC.get(c.streaming_movies, 1),
        CONTRACT.get(c.contract, 0),
        YN.get(c.paperless_billing, 0),
        PAYMENT.get(c.payment_method, 0),
        c.monthly_charges, c.total_charges,
    ]
    return np.array(row, dtype=float).reshape(1, -1)

def risk(p: float) -> str:
    return "High" if p >= 0.70 else "Medium" if p >= 0.40 else "Low"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded":   model is not None,
        "encoder_loaded": label_encoder is not None,
        "features":       feature_cols or "not loaded",
    }

@app.post("/predict")
def predict(c: Customer):
    if model is None:
        raise HTTPException(503, "Model not loaded — copy customer_churn_model.pkl into /models/")
    try:
        X    = encode(c)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        cp   = float(prob[1])
        conf = float(max(prob))
        label = "Yes" if pred == 1 else "No"
        if label_encoder:
            try: label = str(label_encoder.inverse_transform([pred])[0])
            except Exception: pass
        return {
            "churn_prediction":  label,
            "churn_probability": round(cp,   4),
            "stay_probability":  round(1-cp, 4),
            "confidence":        round(conf, 4),
            "risk_level":        risk(cp),
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__ if model else "Not loaded",
        "features":   feature_cols or [],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8001, reload=True)
"""
Standalone Flask App — Customer Churn Predictor
Loads PKL files directly (no FastAPI needed).
Run: python app.py
"""

from flask import Flask, render_template, request, jsonify
import pickle, numpy as np, os

app = Flask(__name__)

# ── Load models ───────────────────────────────────────────────────────────────
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

print(f"[ChurnIQ] Model loaded: {model is not None}")
print(f"[ChurnIQ] Label encoder: {label_encoder is not None}")
print(f"[ChurnIQ] Feature cols: {feature_cols is not None}")

# ── Encoding maps ─────────────────────────────────────────────────────────────
YN       = {"Yes": 1, "No": 0}
GEN      = {"Male": 1, "Female": 0}
INET     = {"DSL": 0, "Fiber optic": 1, "No": 2}
CONTRACT = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT  = {"Electronic check": 0, "Mailed check": 1,
            "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}
ML       = {"No phone service": 0, "No": 1, "Yes": 2}
ISVC     = {"No internet service": 0, "No": 1, "Yes": 2}

def encode(d):
    return np.array([
        GEN.get(d.get("gender", "Male"), 0),
        int(d.get("senior_citizen", 0)),
        YN.get(d.get("partner", "No"), 0),
        YN.get(d.get("dependents", "No"), 0),
        int(d.get("tenure", 0)),
        YN.get(d.get("phone_service", "Yes"), 0),
        ML.get(d.get("multiple_lines", "No"), 1),
        INET.get(d.get("internet_service", "DSL"), 0),
        ISVC.get(d.get("online_security", "No"), 1),
        ISVC.get(d.get("online_backup", "No"), 1),
        ISVC.get(d.get("device_protection", "No"), 1),
        ISVC.get(d.get("tech_support", "No"), 1),
        ISVC.get(d.get("streaming_tv", "No"), 1),
        ISVC.get(d.get("streaming_movies", "No"), 1),
        CONTRACT.get(d.get("contract", "Month-to-month"), 0),
        YN.get(d.get("paperless_billing", "Yes"), 0),
        PAYMENT.get(d.get("payment_method", "Electronic check"), 0),
        float(d.get("monthly_charges", 0)),
        float(d.get("total_charges", 0)),
    ], dtype=float).reshape(1, -1)

def risk(p):
    return "High" if p >= 0.65 else "Medium" if p >= 0.35 else "Low"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded":   model is not None,
        "encoder_loaded": label_encoder is not None,
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check models/ folder."}), 503

    data = request.get_json(force=True)
    try:
        X    = encode(data)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        cp   = float(prob[1])
        conf = float(max(prob))

        label = "Yes" if pred == 1 else "No"
        if label_encoder:
            try:
                label = str(label_encoder.inverse_transform([pred])[0])
            except Exception:
                pass

        return jsonify({
            "churn_prediction":  label,
            "churn_probability": round(cp, 4),
            "stay_probability":  round(1 - cp, 4),
            "confidence":        round(conf, 4),
            "risk_level":        risk(cp),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/model-info")
def model_info():
    return jsonify({
        "model_type": type(model).__name__ if model else "Not loaded",
        "features":   feature_cols or [],
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os


MODEL_FILE = "model.pkl"
VECT_FILE = "vectorizer.pkl"

app = FastAPI(title="Smart Expense Advisor API")

# Load model & vectorizer
if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)
else:
    model = None
    vectorizer = None
    print("⚠️ model.pkl or vectorizer.pkl not found.")


class PredictRequest(BaseModel):
    descriptions: List[str]


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=500,
            detail="Model or vectorizer missing on server."
        )

    cleaned = [d.lower() for d in req.descriptions]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)

    # confidence (if available)
    try:
        confs = model.predict_proba(X).max(axis=1).tolist()
    except Exception:
        confs = [None] * len(preds)

    return {
        "predictions": [
            {
                "description": d,
                "category": p,
                "confidence": float(c) if c else None
            }
            for d, p, c in zip(req.descriptions, preds, confs)
        ]
    }


@app.get("/")
def root():
    return {"message": "Smart Expense Advisor API running"}

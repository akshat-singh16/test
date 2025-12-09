# Replace with full api.py from canvas
"""api.py
Simple FastAPI server that loads model.pkl and vectorizer.pkl and exposes /predict
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from typing import List


MODEL_FILE = os.environ.get('MODEL_FILE', 'model.pkl')
VECT_FILE = os.environ.get('VECT_FILE', 'vectorizer.pkl')


model = None
vectorizer = None


if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECT_FILE)
else:
print('Warning: model or vectorizer not found. Put model.pkl and vectorizer.pkl in the app folder.')


app = FastAPI(title='Smart Expense Advisor API')


class PredictRequest(BaseModel):
descriptions: List[str]


@app.post('/predict')
def predict(req: PredictRequest):
global model, vectorizer
if model is None or vectorizer is None:
raise HTTPException(status_code=500, detail='Model not loaded')
descs = [d.lower() for d in req.descriptions]
X = vectorizer.transform(descs)
preds = model.predict(X)
confidences = None
try:
confidences = model.predict_proba(X).max(axis=1).tolist()
except Exception:
confidences = [None] * len(preds)
return {'predictions': [{'description': d, 'category': p, 'confidence': float(c) if c is not None else None} for d,p,c in zip(req.descriptions, preds, confidences)]}


@app.get('/')
def root():
return {'message': 'Smart Expense Advisor API running'}

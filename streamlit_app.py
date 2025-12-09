# Replace with full streamlit_app.py from canvas
import streamlit as st
import joblib
import os


MODEL_FILE = os.environ.get('MODEL_FILE', 'model.pkl')
VECT_FILE = os.environ.get('VECT_FILE', 'vectorizer.pkl')


st.title('Smart Expense Advisor — Demo')


if not os.path.exists(MODEL_FILE) or not os.path.exists(VECT_FILE):
st.warning('model.pkl or vectorizer.pkl not found in this folder. Please run training first or upload the files.')
else:
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECT_FILE)


desc = st.text_input('Enter transaction description', '')
if st.button('Predict') and desc.strip():
desc_clean = desc.lower()
X = vectorizer.transform([desc_clean])
pred = model.predict(X)[0]
conf = None
try:
conf = model.predict_proba(X).max()
except Exception:
conf = None
st.write('**Predicted category:**', pred)
if conf is not None:
st.write('Confidence:', f"{conf:.2f}")


st.markdown('---')
st.write('You can also upload a CSV with a `description` column for batch predictions (coming soon).')

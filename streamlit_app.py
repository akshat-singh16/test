import streamlit as st
import joblib
import os

MODEL_FILE = "model.pkl"
VECT_FILE = "vectorizer.pkl"

st.title("Smart Expense Advisor — Demo")

# Check if model + vectorizer exist
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECT_FILE):
    st.warning("model.pkl or vectorizer.pkl not found in this folder. Please upload them or train the model.")
else:
    # Load model + vectorizer
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)

    # User input
    description = st.text_input("Enter transaction description", "")

    # Predict button
    if st.button("Predict") and description.strip():
        desc_clean = description.lower()

        # Vectorize
        X = vectorizer.transform([desc_clean])

        # Predict category
        prediction = model.predict(X)[0]

        # Confidence (if model supports predict_proba)
        try:
            confidence = model.predict_proba(X).max()
        except:
            confidence = None

        # Output results
        st.subheader("Prediction Result")
        st.write(f"**Category:** {prediction}")

        if confidence is not None:
            st.write(f"**Confidence:** {confidence:.2f}")
        else:
            st.write("*Confidence score not available for this model.*")

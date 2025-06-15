import streamlit as st
import manual
import upload
import joblib

st.title("Credit Score Prediction")

st.write("This app predicts the credit score of a customer based on their financial information.")
st.write("You can either enter the information manually or upload a CSV file.")

tab1, tab2 = st.tabs(["Upload", "Manual"])

# Load the model
model = joblib.load("cat_model.joblib")
le = joblib.load("label_encoder.pkl")

with tab1:
    upload.run(model, le)

with tab2:
    manual.run(model, le)


import streamlit as st
import joblib
import numpy as np
import os

# ---------------------------
# Load models and scalers
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_smoker = joblib.load(os.path.join(BASE_DIR, "models", "model_smokers.pkl"))
scaler_smoker = joblib.load(os.path.join(BASE_DIR, "models", "scaler_smokers.pkl"))

model_nonsmoker = joblib.load(os.path.join(BASE_DIR, "models", "model_nonsmokers.pkl"))
scaler_nonsmoker = joblib.load(os.path.join(BASE_DIR, "models", "scaler_nonsmokers.pkl"))

model_all = joblib.load(os.path.join(BASE_DIR, "models", "model_all.pkl"))
scaler_all = joblib.load(os.path.join(BASE_DIR, "models", "scaler_all.pkl"))

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")
st.title("💰 Medical Insurance Cost Predictor")
st.write("Fill in the form to predict the expected insurance cost.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 30)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    children = st.number_input("Number of Children", 0, 10, 1)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Optional override for general model
use_general_model = st.checkbox("🔁 Force using general model (all data)", value=False)

# Encode categorical variables
sex_val = 0 if sex == "male" else 1
smoker_val = 1 if smoker == "yes" else 0

region_dict = {
    "northeast": [0, 0, 0],
    "northwest": [1, 0, 0],
    "southeast": [0, 1, 0],
    "southwest": [0, 0, 1]
}
region_encoded = region_dict[region]

# Scale numeric input
numeric_features = np.array([[age, bmi, children]])

# Choose model and scaler
if use_general_model:
    model = model_all
    scaler = scaler_all
    st.write("✅ Using: General model (all data)")
elif smoker == "yes":
    model = model_smoker
    scaler = scaler_smoker
    st.write("✅ Using: Smoker model")
elif smoker == "no":
    model = model_nonsmoker
    scaler = scaler_nonsmoker
    st.write("✅ Using: Non-smoker model")
else:
    model = model_all
    scaler = scaler_all
    st.write("✅ Using: General model (fallback)")

scaled_numeric = scaler.transform(numeric_features)

# Combine all inputs
features = np.concatenate([
    scaled_numeric[0],
    [sex_val, smoker_val],
    region_encoded
]).reshape(1, -1)

# Predict
if st.button("Predict Insurance Charges"):
    prediction = model.predict(features)[0]
    st.success(f"💵 Estimated Insurance Cost: ${prediction:,.2f}")

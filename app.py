import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(page_title="Loan Approval Prediction",
                   page_icon="💰",
                   layout="centered")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        text-align: center;
        margin-top: 20px;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL FILES
# =========================
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# =========================
# TITLE & LOGO
# =========================
st.markdown("<div class='title'>💰 Loan Approval Prediction App</div>", unsafe_allow_html=True)
st.write("Fill in the details below to check your loan approval status.")

# =========================
# USER INPUT FIELDS
# =========================
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", label_encoders["education"].classes_)
self_employed = st.selectbox("Self Employed", label_encoders[" self_employed"].classes_)
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# =========================
# CENTERED PREDICTION BUTTON
# =========================
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("🔍 Predict Loan Approval"):
    # Prepare DataFrame
    input_df = pd.DataFrame([[no_of_dependents, education, self_employed, income_annum,
                              loan_amount, loan_term, cibil_score,
                              residential_assets_value, commercial_assets_value,
                              luxury_assets_value, bank_asset_value]],
                            columns=[" no_of_dependents", "education", " self_employed",
                                     " income_annum", " loan_amount", " loan_term", " cibil_score",
                                     " residential_assets_value", " commercial_assets_value",
                                     " luxury_assets_value", " bank_asset_value"])
    
    # Encode categorical variables
    for col in ["education", " self_employed"]:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])
    
    # Scale numeric columns
    numeric_cols = [' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term',
                    ' cibil_score', ' residential_assets_value', ' commercial_assets_value',
                    ' luxury_assets_value', ' bank_asset_value']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    status = label_encoders[" loan_status"].inverse_transform([prediction])[0]
    
    # Display result
    if status.lower() == "approved":
        st.markdown(f"<div class='success-box'>✅ Loan Status: {status}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='error-box'>❌ Loan Status: {status}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

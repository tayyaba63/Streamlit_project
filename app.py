import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG & CUSTOM CSS
# =========================
st.set_page_config(page_title="Loan Approval App", page_icon="💰", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background: linear-gradient(to bottom right, #f0f4f7, #d9e4ec);
            font-family: 'Segoe UI', sans-serif;
        }
        /* Title styling */
        h1 {
            color: #2c3e50;
            text-align: center;
            font-weight: bold;
        }
        /* Input label style */
        label {
            font-weight: bold !important;
            color: #34495e !important;
        }
        /* Button style */
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            border-radius: 10px;
            height: 3em;
            font-size: 16px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #27ae60;
            color: white;
        }
        /* Result box */
        .success-box {
            background-color: #eafaf1;
            padding: 15px;
            border-radius: 10px;
            color: #2ecc71;
            font-size: 18px;
            text-align: center;
        }
        .error-box {
            background-color: #fdecea;
            padding: 15px;
            border-radius: 10px;
            color: #e74c3c;
            font-size: 18px;
            text-align: center;
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
# TITLE
# =========================
st.title("💰Loan Approval Prediction App")
st.markdown("<hr>", unsafe_allow_html=True)

# =========================
# INPUT FORM WITH COLUMNS
# =========================
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    education = st.selectbox("Education", label_encoders["education"].classes_)
    self_employed = st.selectbox("Self Employed", label_encoders[" self_employed"].classes_)
    income_annum = st.number_input("Annual Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    
with col2:
    loan_term = st.number_input("Loan Term (months)", min_value=0)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# =========================
# PREDICTION
# =========================
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
    
    # Encoding categorical variables
    for col in ["education", " self_employed"]:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])
    
    # Scaling numeric columns
    numeric_cols = [' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term',
                    ' cibil_score', ' residential_assets_value', ' commercial_assets_value',
                    ' luxury_assets_value', ' bank_asset_value']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Prediction
    prediction = model.predict(input_df)[0]
    status = label_encoders[" loan_status"].inverse_transform([prediction])[0]
    
    # Display Result with styled box
    if status.lower() == "approved":
        st.markdown(f"<div class='success-box'>✅ Loan Status: {status}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='error-box'>❌ Loan Status: {status}</div>", unsafe_allow_html=True)

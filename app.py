import streamlit as st
import pandas as pd
import joblib

# ---------------------- Load Saved Models ----------------------
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",  # Loan-related emoji (can replace with PNG file path like "loan_logo.png")
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Custom CSS ----------------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.5em;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
        .approved {
            background-color: #d4edda;
            color: #155724;
        }
        .rejected {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Title ----------------------
st.markdown("<h1 style='text-align: center;'>💰 Loan Approval Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Fill in the details below to check your loan approval status.</p>", unsafe_allow_html=True)

# ---------------------- Sidebar Inputs ----------------------
st.sidebar.header("🔍 Enter Applicant Details")

no_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, step=1)
education = st.sidebar.selectbox("Education", label_encoders["education"].classes_)
self_employed = st.sidebar.selectbox("Self Employed", label_encoders[" self_employed"].classes_)
income_annum = st.sidebar.number_input("Annual Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=0)
cibil_score = st.sidebar.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.sidebar.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0)

# ---------------------- Predict Button ----------------------
if st.sidebar.button("🔮 Predict Loan Approval"):
    # Prepare dataframe
    input_df = pd.DataFrame([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], columns=[
        " no_of_dependents", "education", " self_employed",
        " income_annum", " loan_amount", " loan_term", " cibil_score",
        " residential_assets_value", " commercial_assets_value",
        " luxury_assets_value", " bank_asset_value"
    ])

    # Encoding categorical values
    for col in ["education", " self_employed"]:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Scaling numerical values
    numeric_cols = [
        ' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term',
        ' cibil_score', ' residential_assets_value', ' commercial_assets_value',
        ' luxury_assets_value', ' bank_asset_value'
    ]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Prediction
    prediction = model.predict(input_df)[0]
    status = label_encoders[" loan_status"].inverse_transform([prediction])[0]

    # Display Result
    if status.lower() == "approved":
        st.markdown(f"<div class='result-box approved'> Loan Status: {status}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box rejected'>Loan Status: {status}</div>", unsafe_allow_html=True)

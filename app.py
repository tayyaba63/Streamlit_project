import streamlit as st
import pandas as pd
import joblib

#  Load Save Files
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

#App Title
st.title("💰 Loan Approval Prediction App")
# User Inputs


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

# Button
if st.button("Predict Loan Approval"):
    # Prepare  dataframe
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

    # Encoding
    for col in ["education", " self_employed"]:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Scaling
    numeric_cols = [
        ' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term',
        ' cibil_score', ' residential_assets_value', ' commercial_assets_value',
        ' luxury_assets_value', ' bank_asset_value'
    ]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    #  prediction
    prediction = model.predict(input_df)[0]
    status = label_encoders[" loan_status"].inverse_transform([prediction])[0]

    #  result
    if status.lower() == "approved":
        st.success(f" Loan Status: {status}")
    else:
        st.error(f" Loan Status: {status}")

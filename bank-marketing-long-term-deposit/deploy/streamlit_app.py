import streamlit as st
import requests
import json
import shap
import pickle
from utils import utils
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Website Header
st.set_page_config(page_title="Bank Marketing Prediction App", layout="wide")

# =========================
# Load Artifacts
model_pipeline_path = 'artifacts/xgb_pipeline.pkl'
final_pipeline = utils.load_artifact(model_pipeline_path)

# =========================
# Forms

st.title("🎯 Bank Campaign Acceptance Prediction")
st.markdown("Fill in the information below to estimate the probability of the client accepting the term deposit.")

st.write('Author: Guilherme Garcia Paschoalinoto (www.linkedin.com/in/guilhermegpaschoalinoto/)')
st.divider()

# =========================
# Personal Info
st.subheader("👤 Personal Info")
st.markdown("Provide basic personal information about the client. These features help the model understand demographic patterns.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=38)

with col2:
    job = st.selectbox("Job", [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
        "retired", "self-employed", "services", "student", "technician", 
        "unemployed", "unknown"
    ], index=2)

with col3:
    marital = st.selectbox("Marital Status", ["single", "married", "divorced"], index=0)

with col4:
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"], index=2)


st.divider()

# =========================
# Finance Info
st.subheader("💰 Finance Info")
st.markdown("Provide financial details of the client. These features help the model assess the client's financial capacity and likelihood to subscribe to the term deposit.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    default = st.selectbox("Default?", ["no", "yes"], index=0)

with col2:
    balance = st.number_input("Average Balance", value=243)

with col3:
    housing = st.selectbox("Has Housing Loan?", ["no", "yes"], index=0)

with col4:
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"], index=1)


st.divider()

# =========================
# Campaign Info
st.subheader("📞 Campaign Info")
st.markdown("Provide information about the client's previous interactions with marketing campaigns. These features help the model understand engagement patterns.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"], index=2)

with col2:
    day = st.number_input("Contact Day", min_value=1, max_value=31, value=5)

with col3:
    month = st.selectbox("Contact Month", [
        "jan", "feb", "mar", "apr", "may", "jun", 
        "jul", "aug", "sep", "oct", "nov", "dec"
    ], index=4)

with col4:
    campaign = st.number_input("Contacts in Current Campaign", min_value=1, value=1)

col1, col2, col3, col4 = st.columns(4)

with col1:
    pdays = st.number_input("Days Since Last Contact", value=-1)

with col2:
    previous = st.number_input("Previous Contacts", value=0)

with col3:
    poutcome = st.selectbox("Previous Campaign Outcome", [
        "failure", "success", "other", "unknown"
    ], index=3)

# =========================
# Format payload
st.divider()

input_data = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}

with st.expander("Show input payload"):
    st.json(input_data)

df_input = pd.DataFrame([input_data])

# =========================
# Prediction
if st.button("Predict"):
    try:
        with st.spinner("Sending data to the model..."):
            response = requests.post(
                    "http://localhost:8000/predict",
                    data=json.dumps(input_data),
                    headers={"Content-Type": "application/json"}
                )
        if response.status_code == 200:
            result = response.json()
            y_pred = result.get('y')
            st.header("Prediction Result")

            if y_pred == 1:
                st.success("✅ The model predicts that the client **WILL ACCEPT** the offer.")
            elif y_pred == 0:
                st.warning("⚠️ The model predicts that the client **WILL NOT ACCEPT** the offer.")

            # =========================
            # Feature Importance
            with st.expander("📊 Feature Importance"):
                model = final_pipeline[-1]
                preprocessor = final_pipeline[:-1]

                X_transformed = preprocessor.transform(df_input)
                explainer = shap.Explainer(model, feature_names=preprocessor.get_feature_names_out())
                shap_values = explainer(X_transformed)

                shap.plots.bar(shap_values[0], show=False)
                plt.gcf().set_size_inches(8,4)
                plt.tight_layout()
                st.pyplot(plt.gcf())
                
                feature_importance_df = pd.DataFrame({"feature": shap_values[0].feature_names,
                                                      "shap_value": shap_values[0].values}).sort_values(by="shap_value", key=abs, ascending=False)
                feature_importance_str = feature_importance_df.to_string(index=False)

                st.subheader('LLM Interpretation')
                st.markdown('Response generated by gemini-2.5-flash-lite. The model may make mistakes, so remember to double-check important information.')
                result = utils.llm_explain_feature_importance(feature_importance_str)
                st.info(result.get("explanation", "").strip())
        else:
            st.error(f"Error {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to connect to the server: {e}")
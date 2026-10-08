import sys
import types

# Create a robust unpickling patch for scikit-learn version differences
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except ImportError:
    pass

# Force inject the stub globally into the Python systems environment
if 'sklearn.compose._column_transformer' in sys.modules:
    mod = sys.modules['sklearn.compose._column_transformer']
    if not hasattr(mod, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        setattr(mod, '_RemainderColsList', _RemainderColsList)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

st.set_page_config(
    page_title="Insider Threat Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Insider Threat Detection Dashboard")
st.markdown("---")
st.markdown("### Interactive Threat Evaluation Prototype")
st.write("Input raw user activity metrics below to run the Random Forest model classification pipeline.")

@st.cache_resource
def load_model_pipeline():
    model_dir = "model"
    model_path = os.path.join(model_dir, "random_forest_insider_threat_model_compressed.joblib")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if not os.path.exists(model_path):
        MODEL_URL = "https://github.com"
        with st.spinner("Downloading model binary from secure GitHub release storage... Please wait..."):
            try:
                urllib.request.urlretrieve(MODEL_URL, model_path)
                st.success("Model downloaded successfully!")
            except Exception as download_err:
                st.error(f"Failed to fetch model from release assets: {download_err}")
                return None
                
    try:
        payload = joblib.load(model_path)
        return payload
    except Exception as e:
        st.error(f"Error reading model file: {e}")
        return None

payload = load_model_pipeline()

def add_engineered_features(frame):
    out = frame.copy()

    if {"num_printed_pages_off_hours", "total_printed_pages"}.issubset(out.columns):
        out["off_hours_print_ratio"] = (
            out["num_printed_pages_off_hours"] / (out["total_printed_pages"] + 1)
        )

    if {"total_files_burned", "num_entries"}.issubset(out.columns):
        out["files_burned_per_entry"] = (
            out["total_files_burned"] / (out["num_entries"] + 1)
        )

    available_risk_flags = [
        c for c in ["late_exit_flag", "entry_during_weekend", "is_abroad", "burned_from_other"]
        if c in out.columns
    ]

    if available_risk_flags:
        out["unusual_behavior_count"] = (
            out[available_risk_flags].fillna(0).sum(axis=1)
        )

    if "total_files_burned" in out.columns:
        out["any_file_burn_activity"] = (
            out["total_files_burned"].fillna(0) > 0
        ).astype(int)

    return out

def get_risk_band(probability):
    if probability < 0.25:
        return "🟢 Low"
    elif probability < 0.50:
        return "🟡 Moderate"
    elif probability < 0.75:
        return "🟠 High"
    return "🔴 Very High"

if payload:
    pipeline = payload["model"]
    feature_cols = payload["model_feature_columns"]
    
    st.subheader("📝 Activity Log Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_printed_pages = st.number_input("Total Printed Pages", min_value=0, value=12)
        num_printed_pages_off_hours = st.number_input("Off-Hours Printed Pages", min_value=0, value=2)
        total_files_burned = st.number_input("Total Files Burned (Removable Media)", min_value=0, value=0)
        num_entries = st.number_input("Workplace Access Entries", min_value=0, value=4)
        employee_seniority_years = st.number_input("Employee Seniority (Years)", min_value=0, value=3)

    with col2:
        late_exit_flag = st.selectbox("Late Exit Flag?", ["No", "Yes"])
        entry_during_weekend = st.selectbox("Weekend Access Entry?", ["No", "Yes"])
        is_abroad = st.selectbox("Remote Access from Abroad?", ["No", "Yes"])
        burned_from_other = st.selectbox("Burned Files From Other Profiles?", ["No", "Yes"])
        
        employee_department = st.selectbox(
            "Employee Department", 
            ["Engineering", "IT", "Sales", "Human Resources", "Finance", "Legal"]
        )

    flag_mapping = {"No": 0, "Yes": 1}
    st.markdown("---")
    
    if st.button("🚀 Evaluate Activity Record", type="primary"):
        activity_record = {
            "total_printed_pages": total_printed_pages,
            "num_printed_pages_off_hours": num_printed_pages_off_hours,
            "total_files_burned": total_files_burned,
            "num_entries": num_entries,
            "employee_seniority_years": employee_seniority_years,
            "late_exit_flag": flag_mapping[late_exit_flag],
            "entry_during_weekend": flag_mapping[entry_during_weekend],
            "is_abroad": flag_mapping[is_abroad],
            "burned_from_other": flag_mapping[burned_from_other],
            "employee_department": employee_department
        }
        
        one_row = pd.DataFrame([activity_record])
        one_row = add_engineered_features(one_row)
        one_row = one_row.reindex(columns=feature_cols)
        
        try:
            probability = float(pipeline.predict_proba(one_row)[0, 1])
            predicted_class = int(probability >= 0.50)
            risk_band_string = get_risk_band(probability)
            
            st.subheader("📊 Model Inference Evaluation")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Calculated Threat Probability", value=f"{probability:.2%}")
                st.metric(label="Assigned Severity Band", value=risk_band_string)
                
            with res_col2:
                if predicted_class == 1:
                    st.error("🚨 **POTENTIAL INSIDER THREAT DETECTED**")
                    st.warning("⚠️ **Recommended Action:** Escalate activity record for formal security analyst investigation.")
                else:
                    st.success("✅ **NORMAL ACTIVITY PROFILE**")
                    st.info("ℹ️ **Recommended Action:** No immediate risk anomalies. Continue standard automated monitoring.")
                    
        except Exception as eval_err:
            st.error(f"Inference processing failed. Ensure your feature data schema matches the model criteria. Error: {eval_err}")

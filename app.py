import sys
import types
import os
import urllib.request
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# --- BACKWARD COMPATIBILITY PATCH FOR SKLEARN PIPELINES ---
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, '_RemainderColsList'):
        class _RemainderColsList(list): pass
        _ct._RemainderColsList = _RemainderColsList
except ImportError:
    pass

if 'sklearn.compose._column_transformer' in sys.modules:
    mod = sys.modules['sklearn.compose._column_transformer']
    if not hasattr(mod, '_RemainderColsList'):
        class _RemainderColsList(list): pass
        setattr(mod, '_RemainderColsList', _RemainderColsList)

from sklearn.impute import SimpleImputer
if not hasattr(SimpleImputer, '_fill_dtype'):
    SimpleImputer._fill_dtype = object
# --------------------------------------------------------

st.set_page_config(
    page_title="Insider Threat Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Insider Threat Detection Dashboard")
st.markdown("---")

@st.cache_resource
def load_model_pipeline():
    # Changing the filename guarantees that Streamlit won't reload a corrupted file from previous runs
    model_path = "downloaded_rf_pipeline_v1.joblib"
    
    if not os.path.exists(model_path):
        # ⚠️ GETTING THE CORRECT LINK IS ABSOLUTELY VITAL:
        # Go to your repo -> Click 'Model Binary Artifact' under Releases on the right side.
        # Right-click your actual model file name (ending in .joblib/.pkl) and click "Copy link address"
        MODEL_URL = "https://github.com"
        
        with st.spinner("Downloading 31MB machine learning model artifact from GitHub Releases..."):
            try:
                # Set user-agent headers so GitHub doesn't block the automated request with an HTML response
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                
                urllib.request.urlretrieve(MODEL_URL, model_path)
                st.success("Model binary downloaded and cached locally!")
            except Exception as download_err:
                st.error(f"Failed to fetch model from release assets: {download_err}")
                return None
                
    try:
        payload = joblib.load(model_path)
        return payload
    except Exception as e:
        st.error(f"Error reading model file: {e}")
        st.info("If this keeps failing, your file in the GitHub Releases storage itself might be corrupted or in an incompatible format.")
        return None

payload = load_model_pipeline()

def add_engineered_features(frame):
    out = frame.copy()
    if {"num_printed_pages_off_hours", "total_printed_pages"}.issubset(out.columns):
        out["off_hours_print_ratio"] = out["num_printed_pages_off_hours"] / (out["total_printed_pages"] + 1)
    if {"total_files_burned", "num_entries"}.issubset(out.columns):
        out["files_burned_per_entry"] = out["total_files_burned"] / (out["num_entries"] + 1)

    available_risk_flags = [c for c in ["late_exit_flag", "entry_during_weekend", "is_abroad", "burned_from_other"] if c in out.columns]
    if available_risk_flags:
        out["unusual_behavior_count"] = out[available_risk_flags].fillna(0).sum(axis=1)
    if "total_files_burned" in out.columns:
        out["any_file_burn_activity"] = (out["total_files_burned"].fillna(0) > 0).astype(int)
    return out

def get_risk_band(probability):
    if probability < 0.25: return "🟢 Low"
    elif probability < 0.50: return "🟡 Moderate"
    elif probability < 0.75: return "🟠 High"
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
        employee_department = st.selectbox("Employee Department", ["Engineering", "IT", "Sales", "Human Resources", "Finance", "Legal"])

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
            prob_array = pipeline.predict_proba(one_row)
            probability = float(prob_array[0, 1])
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
                else:
                    st.success("✅ **NORMAL ACTIVITY PROFILE**")
        except Exception as eval_err:
            st.error(f"Inference processing failed: {eval_err}")

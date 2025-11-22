import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="OHCA ITE Predictor", layout="wide")

# --- 1. Helper Function for Query Params ---
def get_param(key, default_value, type_func=float):
    """
    從 URL 取得參數，如果不存在或轉換失敗則回傳預設值。
    支援 st.query_params (Streamlit 新版寫法)
    """
    # 取得 query params
    qp = st.query_params
    
    if key in qp:
        try:
            val = qp[key]
            # 處理布林值字串 (針對 autorun)
            if type_func == bool:
                return val.lower() == 'true'
            return type_func(val)
        except:
            return default_value
    return default_value

# Load models and data
@st.cache_resource
def load_resources():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    model = joblib.load(os.path.join(base_path, 'causal_forest_model.joblib'))
    covariates = joblib.load(os.path.join(base_path, 'covariates.joblib'))
    cutoffs = joblib.load(os.path.join(base_path, 'ite_tertile_cutoffs.joblib'))
    imputer = joblib.load(os.path.join(base_path, 'knn_imputer.joblib'))
    
    return model, covariates, cutoffs, imputer

try:
    model, covariates_list, cutoffs, imputer = load_resources()
    # 註解掉 success 避免畫面太雜
    # st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("OHCA Sodium Bicarbonate ITE Predictor")
st.markdown("""
This application predicts the **Individualized Treatment Effect (ITE)** of Sodium Bicarbonate for OHCA patients.
Please enter the patient's clinical parameters below.
""")

# --- 2. Check for Auto-Run Flag ---
auto_run = get_param('autorun', False, bool)

# Create input form
with st.form("patient_data_form"):
    st.subheader("Patient Demographics & Prehospital Data")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 使用 get_param 設定預設值
        def_age = get_param('age', 65, int)
        age = st.number_input("Age", min_value=0, max_value=120, value=def_age)
        
        def_sex = get_param('sex', 1, int) # Default 1 (Male)
        sex = st.selectbox("Sex", options=[0, 1], index=def_sex, format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
        
        def_place = get_param('place_new', 0, int)
        place_new = st.number_input("Location Category (place_new)", value=def_place)
        
        def_witnessed = get_param('witnessed_core', 0, int)
        witnessed_core = st.selectbox("Witnessed", options=[0, 1], index=def_witnessed, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        
        def_bystander = get_param('bystander_core', 0, int)
        bystander_core = st.selectbox("Bystander CPR", options=[0, 1], index=def_bystander, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

    with col2:
        def_resp = get_param('responsetime', 5.0, float)
        responsetime = st.number_input("Response Time (min)", min_value=0.0, value=def_resp)
        
        def_scene = get_param('scenetohosptime', 15.0, float)
        scenetohosptime = st.number_input("Scene to Hospital Time (min)", min_value=0.0, value=def_scene)
        
        def_aed = get_param('aed_core', 0, int)
        aed_core = st.selectbox("AED Used", options=[0, 1], index=def_aed, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        
        def_airway = get_param('airway', 0, int)
        airway = st.number_input("Airway Management", value=def_airway)
        
        def_bosmin = get_param('bosmin_core', 0, int)
        bosmin_core = st.selectbox("Prehospital Epinephrine (bosmin)", options=[0, 1], index=def_bosmin, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

    st.subheader("Clinical & Lab Data")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        def_rhythm = get_param('initialrhythm_core', 0, int)
        initialrhythm_core = st.number_input("Initial Rhythm", value=def_rhythm)
        
        def_etco2 = get_param('etco2_core', 0.0, float)
        etco2_core = st.number_input("EtCO2", value=def_etco2)
        
        def_lactic = get_param('lactic', 0.0, float)
        lactic = st.number_input("Lactate (mmol/L)", value=def_lactic)
        
        def_ph = get_param('ph', 7.0, float)
        ph = st.number_input("pH", value=def_ph, format="%.2f")
        
    with col6:
        def_hco3 = get_param('hco3', 20.0, float)
        hco3 = st.number_input("HCO3 (mmol/L)", value=def_hco3)
        
        def_pco2 = get_param('pco2', 40.0, float)
        pco2 = st.number_input("pCO2 (mmHg)", value=def_pco2)
        
        def_be = get_param('be', 0.0, float)
        be = st.number_input("Base Excess", value=def_be)
        
    with col7:
        def_cre = get_param('cre', 1.0, float)
        cre = st.number_input("Creatinine (mg/dL)", value=def_cre)
        
        def_na = get_param('na', 140.0, float)
        na = st.number_input("Sodium (mmol/L)", value=def_na)
        
        def_k = get_param('k', 4.0, float)
        k = st.number_input("Potassium (mmol/L)", value=def_k)

    # Button logic
    submitted = st.form_submit_button("Predict ITE")

# --- 3. Trigger Prediction (Button OR Auto-run) ---
if submitted or auto_run:
    # Create DataFrame
    input_data = {
        'age': [age],
        'sex': [sex],
        'responsetime': [responsetime],
        'scenetohosptime': [scenetohosptime],
        'place_new': [place_new],
        'witnessed_core': [witnessed_core],
        'bystander_core': [bystander_core],
        'aed_core': [aed_core],
        'airway': [airway],
        'bosmin_core': [bosmin_core],
        'initialrhythm_core': [initialrhythm_core],
        'lactic': [lactic],
        'ph': [ph],
        'hco3': [hco3],
        'pco2': [pco2],
        'be': [be],



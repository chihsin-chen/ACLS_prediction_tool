import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="OHCA ITE Predictor", layout="wide")

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
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("OHCA Sodium Bicarbonate ITE Predictor")
st.markdown("""
This application predicts the **Individualized Treatment Effect (ITE)** of Sodium Bicarbonate for OHCA patients.
Please enter the patient's clinical parameters below.
""")

# Create input form
with st.form("patient_data_form"):
    st.subheader("Patient Demographics & Prehospital Data")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=65)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
        place_new = st.number_input("Location Category (place_new)", value=0)
        witnessed_core = st.selectbox("Witnessed", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        bystander_core = st.selectbox("Bystander CPR", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

    with col2:
        responsetime = st.number_input("Response Time (min)", min_value=0.0, value=5.0)
        scenetohosptime = st.number_input("Scene to Hospital Time (min)", min_value=0.0, value=15.0)
        aed_core = st.selectbox("AED Used", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        airway = st.number_input("Airway Management", value=0)
        bosmin_core = st.selectbox("Prehospital Epinephrine (bosmin)", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

    st.subheader("Clinical & Lab Data")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        initialrhythm_core = st.number_input("Initial Rhythm", value=0)
        etco2_core = st.number_input("EtCO2", value=0.0)
        lactic = st.number_input("Lactate (mmol/L)", value=0.0)
        ph = st.number_input("pH", value=7.0, format="%.2f")
        
    with col6:
        hco3 = st.number_input("HCO3 (mmol/L)", value=20.0)
        pco2 = st.number_input("pCO2 (mmHg)", value=40.0)
        be = st.number_input("Base Excess", value=0.0)
        
    with col7:
        cre = st.number_input("Creatinine (mg/dL)", value=1.0)
        na = st.number_input("Sodium (mmol/L)", value=140.0)
        k = st.number_input("Potassium (mmol/L)", value=4.0)

    # Hidden inputs or defaults for missing covariates if any? 
    # The list has 20 items. Let's verify we have all.
    # ['age', 'sex', 'responsetime', 'scenetohosptime', 'place_new', 'witnessed_core', 'bystander_core', 'aed_core', 'airway', 'bosmin_core', 'initialrhythm_core', 'lactic', 'ph', 'hco3', 'pco2', 'be', 'cre', 'na', 'k', 'etco2_core']
    # We have:
    # 1. age, 2. sex, 3. responsetime, 4. scenetohosptime, 5. place_new, 6. witnessed_core, 7. bystander_core, 
    # 8. aed_core, 9. airway, 10. bosmin_core, 11. initialrhythm_core, 12. lactic, 13. ph, 14. hco3, 
    # 15. pco2, 16. be, 17. cre, 18. na, 19. k, 20. etco2_core
    # All 20 are present.

    submitted = st.form_submit_button("Predict ITE")

if submitted:
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
        'cre': [cre],
        'na': [na],
        'k': [k],
        'etco2_core': [etco2_core]
    }
    
    df = pd.DataFrame(input_data)
    
    # Ensure column order matches covariates
    df = df[covariates_list]
    
    # Impute missing values (although inputs are filled, this handles any potential issues or if we allow NaNs later)
    # The imputer returns a numpy array, we need to convert back to DF if model expects DF, or just use array if model accepts it.
    # CausalForestDML usually accepts arrays or DFs.
    X_imputed = imputer.transform(df)
    
    # Predict
    try:
        # model.effect(X) returns the CATE (Conditional Average Treatment Effect)
        ite_pred = model.effect(X_imputed)[0]
        ite_percent = ite_pred * 100
        
        st.divider()
        st.header("Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(label="Predicted ITE", value=f"{ite_percent:.2f}%")
            
        # Determine recommendation based on cutoffs
        # Cutoffs: [-0.27, -0.04, 0.06, 0.25]
        # Lower: < -0.04
        # Middle: -0.04 to 0.06
        # Upper: > 0.06
        
        lower_bound = cutoffs[1] # -0.04
        upper_bound = cutoffs[2] # 0.06
        
        with col_res2:
            if ite_pred > upper_bound:
                st.success("Recommendation: **Give Sodium Bicarbonate** (Upper Tertile)")
                st.markdown(f"ITE > {upper_bound*100:.1f}%")
            elif ite_pred < lower_bound:
                st.error("Recommendation: **Do Not Give Sodium Bicarbonate** (Lower Tertile)")
                st.markdown(f"ITE < {lower_bound*100:.1f}%")
            else:
                st.warning("Recommendation: **No Significant Difference** (Middle Tertile)")
                st.markdown(f"{lower_bound*100:.1f}% <= ITE <= {upper_bound*100:.1f}%")
                
    except Exception as e:
        st.error(f"Prediction failed: {e}")


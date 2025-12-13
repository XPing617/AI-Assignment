import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Heart Health AI", page_icon="🫀", layout="wide")

st.markdown("""
<style>
.main { background-color: #f5f7f9; }
div.stButton > button {
    width: 100%; background-color: #FF4B4B; color: white;
    height: 3em; font-weight: bold; font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- 2. DATA & MODEL LOADING ---
@st.cache_data
def load_and_train():
    # Load data
    df = pd.read_csv('heart.csv').drop_duplicates()

    # X/y Split
    X = df.drop('target', axis=1)
    y = df['target']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model Training
    # class_weight='balanced' helps if data is uneven, though this dataset is roughly balanced.
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_train.columns

model, model_accuracy, feature_order = load_and_train()

# --- 3. HEADER ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
with col2:
    st.title("Cardio Care AI")
    st.markdown("### Professional Heart Disease Risk Assessment")

st.markdown("---")

# --- 4. INPUT FORM ---
st.subheader("📝 Patient Information")

col1, col2, col3 = st.columns(3)
with col1:
    name = st.text_input("Patient Name", "Jane Doe")
with col2:
    age = st.number_input("Age", 20, 90, 50)
with col3:
    # Data: 1=Male, 0=Female
    sex = st.selectbox("Sex", (1, 0), format_func=lambda x: "Male" if x == 1 else "Female")

with st.expander("🩺 Vitals & Blood Work", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        trestbps = st.number_input("Blood Pressure (mm Hg)", 80, 200, 120, help="Resting Blood Pressure")
    with c2:
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    with c3:
        # Data: 1=True, 0=False
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl?",
            (0, 1),
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
    with c4:
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)

with st.expander("🫀 Heart Exam & Symptoms", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        # Data: 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic
        cp = st.selectbox(
            "Chest Pain Type",
            (0, 1, 2, 3),
            format_func=lambda x: [
                "Typical Angina (Most Severe)",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ][x]
        )
        # Data: 1=Yes, 0=No
        exang = st.selectbox("Exercise Induced Angina?", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")

    with c2:
        # Data: 0, 1, 2
        restecg = st.selectbox(
            "Resting ECG Result",
            (0, 1, 2),
            format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x]
        )
        # Data Correction: 2=Upsloping (Healthy), 1=Flat, 0=Downsloping
        slope_label = st.selectbox(
            "ST Slope",
            ("Upsloping", "Flat", "Downsloping")
        )
        slope_map = {"Upsloping": 2, "Flat": 1, "Downsloping": 0}
        slope = slope_map[slope_label]

    with c3:
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0, step=0.1, format="%.2f")
        ca = st.slider("Major Vessels (Fluoroscopy)", 0, 3, 0)

        # Data Correction: 2=Normal, 1=Fixed, 3=Reversible
        thal_label = st.selectbox(
            "Thalassemia",
            ("Normal", "Fixed Defect", "Reversible Defect")
        )
        thal_map = {"Normal": 2, "Fixed Defect": 1, "Reversible Defect": 3}
        thal = thal_map[thal_label]

# --- 5. INPUT DATAFRAME ---
# Ensure columns match the exact training order
input_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol,
    fbs, restecg, thalach, exang,
    oldpeak, slope, ca, thal
]], columns=feature_order)

st.markdown("---")

# --- 6. REPORT & PREDICTION ---
st.subheader(f"📊 Assessment Summary for {name}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Blood Pressure", f"{trestbps} mm Hg")
m2.metric("Cholesterol", f"{chol} mg/dl")
m3.metric("Max Heart Rate", f"{thalach} bpm")
m4.metric("ST Depression", f"{oldpeak:.2f}")

st.caption(f"Model Accuracy (Test Set): {model_accuracy:.2%}")

if st.button("RUN DIAGNOSTIC MODEL"):
    # CRITICAL LOGIC FIX: 
    # In this dataset, Target 0 = Disease, Target 1 = Healthy.
    # We want the probability of Class 0 (Disease).
    risk_prob = model.predict_proba(input_data)[0][0]

    # Threshold 0.5 (50%)
    if risk_prob >= 0.5:
        st.error("⚠️ HIGH RISK DETECTED")
        st.markdown(f"""
        <div style='background-color:#ffe6e6; padding:20px; border-radius:10px; border-left: 5px solid #ff4b4b;'>
        <h3 style='color:#b30000;'>Diagnosis: Potential Heart Disease</h3>
        <p><strong>Estimated Risk Score:</strong> {risk_prob*100:.1f}%</p>
        <p>The model has identified patterns consistent with heart disease. Please consult a cardiologist for further testing.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ LOW RISK / HEALTHY")
        st.markdown(f"""
        <div style='background-color:#e6fffa; padding:20px; border-radius:10px; border-left: 5px solid #00cc99;'>
        <h3 style='color:#00664d;'>Diagnosis: Healthy Profile</h3>
        <p><strong>Estimated Risk Score:</strong> {risk_prob*100:.1f}%</p>
        <p>The model predicts a low probability of heart disease. Continue maintaining a healthy lifestyle.</p>
        </div>
        """, unsafe_allow_html=True)


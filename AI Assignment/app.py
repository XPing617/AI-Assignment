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
    df = pd.read_csv('heart.csv').drop_duplicates()

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest DOES NOT require scaling
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
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
    sex = st.selectbox("Sex", (1, 0), format_func=lambda x: "Male" if x == 1 else "Female")

with st.expander("🩺 Vitals & Blood Work", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        trestbps = st.number_input("Blood Pressure (mm Hg)", 80, 200, 120)
    with c2:
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    with c3:
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
        cp = st.selectbox(
            "Chest Pain Type",
            (0, 1, 2, 3),
            format_func=lambda x: [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ][x]
        )
        exang = st.selectbox("Exercise Induced Angina?", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")

    with c2:
        restecg = st.selectbox(
            "Resting ECG Result",
            (0, 1, 2),
            format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x]
        )
        slope = st.selectbox(
            "ST Slope",
            (0, 1, 2),
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x]
        )

    with c3:
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0, step=0.1, format="%.2f")
        ca = st.slider("Major Vessels (Fluoroscopy)", 0, 3, 0)

        thal_label = st.selectbox(
            "Thalassemia",
            ("Normal", "Fixed Defect", "Reversible Defect")
        )
        thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
        thal = thal_map[thal_label]

# --- 5. INPUT DATAFRAME ---
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

st.caption(f"Model Accuracy (Test Set): {model_accuracy:.2f}")

if st.button("RUN DIAGNOSTIC MODEL"):
    risk_prob = model.predict_proba(input_data)[0][1]

    if risk_prob >= 0.5:
        st.error("⚠️ HIGH RISK DETECTED")
        st.markdown(f"""
        <div style='background-color:#ffcccc; padding:20px; border-radius:10px;'>
        <h3>Diagnosis: Positive for Heart Disease</h3>
        <p><strong>Estimated Risk:</strong> {risk_prob*100:.1f}%</p>
        <p>Please consult a cardiologist.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ LOW RISK / HEALTHY")
        st.markdown(f"""
        <div style='background-color:#d4edda; padding:20px; border-radius:10px;'>
        <h3>Diagnosis: Negative (Healthy)</h3>
        <p><strong>Estimated Risk:</strong> {(1-risk_prob)*100:.1f}%</p>
        <p>Maintain a healthy lifestyle.</p>
        </div>
        """, unsafe_allow_html=True)

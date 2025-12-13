import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Heart Health AI", page_icon="🫀", layout="wide")

# Custom CSS to make it look nicer
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    div.stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-weight: bold;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & MODEL LOADING ---
@st.cache_data
def load_and_train():
    df = pd.read_csv('heart.csv').drop_duplicates()
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train_scaled, y_train)
    
    return rfc, scaler

model, scaler = load_and_train()

# --- 3. HEADER ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100) # Simple heart icon
with col2:
    st.title("Cardio Care AI")
    st.markdown("### Professional Heart Disease Risk Assessment")

st.markdown("---")

# --- 4. INPUT FORM ---

with st.container():
    st.subheader("📝 Patient Information")
    
    # GROUP 1: DEMOGRAPHICS
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Patient Name", "Jane Doe")
    with col2:
        age = st.number_input("Age", 20, 90, 50)
    with col3:
        sex = st.selectbox("Sex", (1, 0), format_func=lambda x: "Male" if x == 1 else "Female")

    # GROUP 2: VITALS
    with st.expander("🩺 Vitals & Blood Work (Click to Expand)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            trestbps = st.number_input("Blood Pressure (mm Hg)", 80, 200, 120, help="Resting Blood Pressure")
        with c2:
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        with c3:
            fbs = st.selectbox("Fasting Blood Sugar > 120?", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")
        with c4:
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)

    # GROUP 3: SYMPTOMS & TESTS
    with st.expander("🫀 Heart Exam & Symptoms (Click to Expand)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            cp = st.selectbox("Chest Pain Type", (0, 1, 2, 3), 
                              format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
            exang = st.selectbox("Exercise Induced Angina?", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")
        with c2:
            restecg = st.selectbox("Resting ECG Result", (0, 1, 2), 
                                   format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
            slope = st.selectbox("ST Slope", (0, 1, 2), format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        with c3:
            # We use format="%.2f" here to ensure the input box shows 1.10 instead of 1.0999
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0, step=0.1, format="%.2f")
            ca = st.slider("Major Vessels (Fluoroscopy)", 0, 3, 0)
            thal = st.selectbox("Thalassemia", (0, 1, 2, 3), format_func=lambda x: ["Null", "Fixed Defect", "Normal", "Reversable"][x])

# Data Preprocessing
input_data = pd.DataFrame({
    'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
    'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
    'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
})

st.markdown("---")

# --- 5. REPORT SUMMARY & PREDICTION ---
st.subheader(f"📊 Assessment Summary for {name}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Blood Pressure", f"{trestbps} mm Hg", delta_color="inverse")
m2.metric("Cholesterol", f"{chol} mg/dl", delta_color="inverse")
m3.metric("Max Heart Rate", f"{thalach} bpm")
# Showing ST Depression formatted nicely to 2 decimal places (e.g., 1.10)
m4.metric("ST Depression", f"{oldpeak:.2f}") 

st.markdown("<br>", unsafe_allow_html=True) 

# PREDICT BUTTON
if st.button("RUN DIAGNOSTIC MODEL"):
    # Scale Input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    
    # DISPLAY RESULT (Simplified)
    if prediction[0] == 1:
        st.error(f"⚠️ **HIGH RISK DETECTED**")
        st.markdown(f"""
            <div style="background-color: #ffcccc; padding: 20px; border-radius: 10px; border-left: 5px solid #ff0000;">
                <h3 style="color: #990000;">Diagnosis: Positive for Heart Disease</h3>
                <p>The model has detected patterns consistent with heart disease.</p>
                <ul>
                    <li><strong>Action Required:</strong> Please schedule a cardiology consultation immediately.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ **LOW RISK / HEALTHY**")
        st.markdown(f"""
            <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                <h3 style="color: #155724;">Diagnosis: Negative (Healthy)</h3>
                <p>The model has not detected significant signs of heart disease.</p>
                <ul>
                    <li><strong>Action Required:</strong> Maintain healthy lifestyle and routine checkups.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

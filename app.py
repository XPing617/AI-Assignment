import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('heart.csv')
    df = df.drop_duplicates()
    return df

df = load_data()

# Prepare Data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models (We train both to show comparison)
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_scaled, y_train)

svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

# --- 2. SIDEBAR (User Inputs) ---
st.sidebar.header("Patient Input Data")

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex', (1, 0), format_func=lambda x: 'Male' if x == 1 else 'Female')
    cp = st.sidebar.selectbox('Chest Pain Type', (0, 1, 2, 3), format_func=lambda x: f"Type {x}")
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 130)
    chol = st.sidebar.slider('Cholesterol', 126, 564, 246)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (1, 0), format_func=lambda x: 'True' if x == 1 else 'False')
    restecg = st.sidebar.selectbox('Resting ECG', (0, 1, 2))
    thalach = st.sidebar.slider('Max Heart Rate', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST', (0, 1, 2))
    ca = st.sidebar.slider('Major Vessels (0-3)', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia', (0, 1, 2, 3))

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 3. MAIN PAGE ---
st.title("❤️ Heart Disease Prediction System")
st.write("Use the sidebar to input patient details and get a prediction.")

# Tabs for different sections
tab1, tab2 = st.tabs(["🏥 Prediction Demo", "📊 Model Comparison"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.subheader("Patient Report")
    st.dataframe(input_df)

    # Scale the user input
    input_scaled = scaler.transform(input_df)

    # Prediction Button
    if st.button("Predict Results"):
        # Use Random Forest for the main prediction (usually more robust)
        prediction = rfc.predict(input_scaled)
        prob = rfc.predict_proba(input_scaled)

        st.subheader("Diagnosis:")
        if prediction[0] == 1:
            st.error(f"⚠️ POSITIVE: High likelihood of Heart Disease detected.")
            st.write(f"Confidence: {prob[0][1] * 100:.2f}%")
        else:
            st.success(f"✅ NEGATIVE: Patient appears healthy.")
            st.write(f"Confidence: {prob[0][0] * 100:.2f}%")

# --- TAB 2: MODEL COMPARISON (Charts) ---
with tab2:
    st.header("Random Forest vs. SVM Performance")
    
    # 1. Comparison Metrics
    rfc_acc = accuracy_score(y_test, rfc.predict(X_test_scaled))
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    
    col1, col2 = st.columns(2)
    col1.metric("Random Forest Accuracy", f"{rfc_acc:.2%}")
    col2.metric("SVM Accuracy", f"{svm_acc:.2%}")

    # 2. Confusion Matrix Plot
    st.subheader("Confusion Matrix Comparison")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # RFC Matrix
    sns.heatmap(confusion_matrix(y_test, rfc.predict(X_test_scaled)), annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax[0])
    ax[0].set_title("Random Forest")
    
    # SVM Matrix
    sns.heatmap(confusion_matrix(y_test, svm.predict(X_test_scaled)), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[1])
    ax[1].set_title("SVM")
    
    st.pyplot(fig)
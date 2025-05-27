# app.py
# app.py

import streamlit as st
import numpy as np
import pickle

# تحميل النموذج
with open("diabetes_xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# إعداد الصفحة
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# عنوان الصفحة
st.title("🩺 Diabetes Prediction App")
st.markdown("أدخل معلوماتك الصحية لتقدير احتمال إصابتك بالسكري.")

# واجهة الإدخال
col1, col2 = st.columns(2)

with col1:
    HighBP = int(st.selectbox("High Blood Pressure", [False, True]))
    HighChol = int(st.selectbox("High Cholesterol", [False, True]))
    CholCheck = int(st.selectbox("Cholesterol Check", [False, True]))
    BMI = st.slider("BMI", 10.0, 50.0, 25.0)
    Smoker = int(st.selectbox("Smoker", [False, True]))
    Stroke = int(st.selectbox("Stroke History", [False, True]))
    GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

with col2:
    DiffWalk = int(st.selectbox("Difficulty Walking", [False, True]))
    Age = st.slider("Age", 10, 90, 30)
    PhysActivity = int(st.selectbox("Physical Activity", [False, True]))
    Fruits = int(st.selectbox("Eat Fruits", [False, True]))
    Veggies = int(st.selectbox("Eat Vegetables", [False, True]))
    HvyAlcoholConsump = int(st.selectbox("Heavy Alcohol Consumption", [False, True]))
    HeartDiseaseorAttack = int(st.selectbox("Heart Disease or Attack", [False, True]))
    PhysHlth = st.slider("Physical Health (days)", 0, 30, 0)

# نموذج الإدخال
features = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                      PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, PhysHlth, DiffWalk, Age]])

# زر التنبؤ
if st.button("🔮 Predict"):
    prediction = model.predict(features)

    if prediction[0] == 0:
        st.success("✅ You are not diabetic.")
        st.image("normal.png", caption="Normal Blood Sugar Reading")

    else:
        if Age < 20:
            st.error("⚠️ High risk of Type 1 Diabetes (age < 20).")
            st.markdown("🧠 Type 1 usually affects younger people.")
            st.image("type1.jpg", caption="Likely Type 1 Diabetes")
        else:
            st.warning("⚠️ High risk of Type 2 Diabetes (age ≥ 20).")
            st.markdown("🧠 Type 2 is more common among adults (~90%).")
            st.image("type2.png", caption="Likely Type 2 Diabetes")

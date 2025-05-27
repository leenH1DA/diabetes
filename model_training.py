import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import base64


@st.cache_data  # لتخزين نتائج التدريب وعدم تكرارها
def load_and_train_model():
    csv_file = "diabetes_012_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(csv_file)

    selected_features = ['GenHlth', 'PhysHlth', 'Income', 'MentHlth', 'DiffWalk', 'BMI', 'HighBP']

    df["Diabetes_binary"] = df["Diabetes_012"].replace({1:1, 2:1})

    X = df[selected_features]
    y = df["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    with open("diabetes_xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler


try:
    model = pickle.load(open("diabetes_xgb_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    model, scaler = load_and_train_model()

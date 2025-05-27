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

# ===============================
# تحميل البيانات وتدريب النموذج
# ===============================

@st.cache_data  # لتخزين نتائج التحميل والتدريب وعدم تكرارها
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

# ===============================
# تحميل أو تدريب الموديل
# ===============================
try:
    model = pickle.load(open("diabetes_xgb_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    model, scaler = load_and_train_model()

# ===============================
# واجهة Streamlit التفاعلية
# ===============================

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# قراءة صورة الخلفية وتحويلها إلى base64
with open("back.png", "rb") as f:
    image_data = f.read()
background_image = base64.b64encode(image_data).decode()

# تطبيق خلفية كاملة للصفحة مع بعض تحسينات النصوص والأزرار
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{background_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
        color: #000033;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    h1, h2, h3 {{
        color: #003366;  /* أزرق داكن */
        text-shadow: 1px 1px 2px #ffffff;
        
    }}
    div[data-testid="stSelectbox"], div[data-testid="stSlider"], div[data-testid="stNumberInput"] {{
        background-color: rgba(204, 255, 221, 0.8); /* أخضر فاتح مع شفافية */
        padding: 10px;
        border-radius: 10px;
    }}
    div.stButton > button {{
        background-color: #81c784;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        padding: 8px 16px;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    div.stButton > button:hover {{
        background-color: #66bb6a;
        color: white;
    }}
    .stAlert {{
        border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# إعداد الخلفية والصياغة
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{background_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* عنوان بخلفية */
    .custom-title {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 16px 24px;
        border-radius: 15px;
        color: #003366;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-top: 20px;
    }}

    /* جملة توضيحية بخلفية */
    .custom-subtitle {{
        background-color: rgba(255, 255, 255, 0.75);
        padding: 12px 20px;
        border-radius: 12px;
        color: #000;
        font-size: 18px;
        text-align: center;
        max-width: 700px;
        margin: 10px auto 30px auto;
        box-shadow: 0 3px 6px rgba(0,0,0,0.15);
    }}
    </style>
""", unsafe_allow_html=True)

# استخدم markdown لعرض العنوان والنص بخلفية
st.markdown('<div class="custom-title">🩺 Welcome to the Diabetes Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">Please enter your health information below to predict the risk of diabetes.</div>', unsafe_allow_html=True)

# === إدخال المستخدم ===
col1, col2 = st.columns(2)

with col1:
    GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
    PhysHlth = st.slider("Physical Health (days of poor health)", 0, 30, 0)
    Income = st.selectbox("Income Level (1=Lowest, 8=Highest)", list(range(1, 9)), index=3)
    MentHlth = st.slider("Mental Health (days of poor health)", 0, 30, 0)

with col2:
    DiffWalk = int(st.selectbox("Difficulty Walking", [False, True]))
    BMI = st.slider("BMI", 10.0, 50.0, 25.0)
    HighBP = int(st.selectbox("High Blood Pressure", [False, True]))
    Age = st.slider("Age", 10, 90, 30)

features = np.array([[GenHlth, PhysHlth, Income, MentHlth, DiffWalk, BMI, HighBP]])
features_scaled = scaler.transform(features)

if st.button("🔮 Predict"):
    prediction = model.predict(features_scaled)[0]

    if prediction == 0:
        st.success("✅ You are predicted NOT to have diabetes.")
        st.image("normal.png", caption="Normal Blood Sugar Reading")
    else:
        if Age < 20:
            st.error("⚠️ High risk of Type 1 Diabetes (Age < 20).")
            st.image("type1.jpg", caption="Type 1 Diabetes Indicator")
        else:
            st.warning("⚠️ High risk of Type 2 Diabetes (Age ≥ 20).")
            st.image("type2.png", caption="Type 2 Diabetes Indicator")

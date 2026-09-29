
import base64
import os
import textwrap
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Health Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# FILE NAMES
# ============================================================

MODEL_FILE = "heart_disease_rf.pkl"
FEATURE_FILE = "feature_columns.pkl"
IMAGE_FILE = "WhatsApp Image 2026-07-30 at 23.30.29.jpeg"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(MODEL_FILE):
    st.error(f"Model file not found: {MODEL_FILE}")
    st.stop()

if not os.path.exists(FEATURE_FILE):
    st.error(f"Feature file not found: {FEATURE_FILE}")
    st.stop()


# ============================================================
# LOAD MODEL AND FEATURES
# ============================================================

try:
    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURE_FILE)
except Exception as error:
    st.error("Unable to load the model or feature file.")
    st.code(str(error))
    st.stop()


# ============================================================
# LOAD PROFILE IMAGE
# ============================================================

def get_image_base64(image_path):
    if not os.path.exists(image_path):
        return None

    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception:
        return None


profile_image = get_image_base64(IMAGE_FILE)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    textwrap.dedent("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef8ff 0%, #f8fbff 50%, #e7f4ff 100%);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .header-box {
        background: linear-gradient(135deg, #075985, #0284c7, #38bdf8);
        padding: 42px 30px;
        border-radius: 24px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 12px 30px rgba(3, 105, 161, 0.22);
    }

    .header-title {
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        color: white;
    }

    .header-subtitle {
        font-size: 18px;
        margin: 10px 0 5px;
        color: white;
    }

    .header-small {
        font-size: 15px;
        margin: 0;
        color: #e0f2fe;
    }

    .intro-box {
        background: white;
        padding: 25px 28px;
        border-radius: 20px;
        margin-bottom: 25px;
        border: 1px solid #dbeafe;
        box-shadow: 0 7px 22px rgba(15, 23, 42, 0.06);
    }

    .intro-title {
        color: #075985;
        font-size: 25px;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .intro-text {
        color: #475569;
        line-height: 1.7;
        font-size: 16px;
        margin: 5px 0;
    }

    .section-header {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        padding: 17px 20px;
        border-radius: 15px;
        margin-bottom: 18px;
        border-left: 5px solid #0284c7;
    }

    .section-title {
        margin: 0;
        color: #075985;
        font-size: 22px;
        font-weight: 750;
    }

    .section-subtitle {
        margin: 5px 0 0;
        color: #64748b;
        font-size: 14px;
    }

    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #075985, #0284c7);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 14px 20px;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 7px 18px rgba(2, 132, 199, 0.20);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1, #0ea5e9);
        color: white;
    }

    .result-box {
        padding: 30px;
        border-radius: 22px;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 25px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }

    .result-risk {
        background: #fff1f2;
        border: 2px solid #fb7185;
    }

    .result-safe {
        background: #f0fdf4;
        border: 2px solid #4ade80;
    }

    .result-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .percentage {
        font-size: 34px;
        font-weight: 800;
        color: #075985;
    }

    .result-description {
        color: #64748b;
        margin-top: 8px;
    }

    .info-card {
        background: white;
        padding: 28px;
        border-radius: 22px;
        margin-top: 30px;
        border: 1px solid #dbeafe;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
    }

    .info-title {
        color: #075985;
        font-size: 27px;
        font-weight: 800;
        margin-bottom: 15px;
    }

    .info-text {
        color: #475569;
        line-height: 1.8;
        font-size: 15px;
        margin: 8px 0;
    }

    .developer-card {
        background: white;
        padding: 32px;
        border-radius: 22px;
        text-align: center;
        margin-top: 30px;
        border: 1px solid #dbeafe;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.07);
    }

    .developer-image {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        object-fit: cover;
        border: 5px solid #0284c7;
        margin-bottom: 15px;
    }

    .developer-name {
        font-size: 25px;
        font-weight: 800;
        color: #075985;
        margin-top: 5px;
    }

    .developer-role {
        color: #64748b;
        font-size: 16px;
        margin: 5px 0;
    }

    .footer-box {
        text-align: center;
        margin-top: 40px;
        padding: 25px 15px;
        color: #64748b;
        border-top: 1px solid #cbd5e1;
        font-size: 14px;
        line-height: 1.8;
    }

    .footer-title {
        color: #075985;
        font-weight: 800;
        font-size: 17px;
    }

    @media (max-width: 768px) {
        .header-title { font-size: 30px; }
        .header-subtitle { font-size: 15px; }
        .intro-title { font-size: 21px; }
        .section-title { font-size: 19px; }
    }
    </style>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    textwrap.dedent("""
    <div class="header-box">
        <div class="header-title">
            ❤️ Heart Health Predictor
        </div>
        <div class="header-subtitle">
            Machine Learning Based Heart Disease Risk Prediction System
        </div>
        <div class="header-small">
            Powered by Random Forest Classification
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    textwrap.dedent("""
    <div class="intro-box">
        <div class="intro-title">
            🩺 Welcome to Heart Health Predictor
        </div>
        <p class="intro-text">
            This application uses a trained Machine Learning model
            to estimate the likelihood of heart disease based on
            cardiovascular and clinical parameters.
        </p>
        <p class="intro-text">
            Enter the patient's information below and click
            <b>Check Heart Risk</b> to generate the prediction.
        </p>
    </div>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# INPUT COLUMNS
# ============================================================

left_column, right_column = st.columns(2, gap="large")


# ============================================================
# PATIENT INFORMATION
# ============================================================

with left_column:
    st.markdown(
        textwrap.dedent("""
        <div class="section-header">
            <div class="section-title">
                👤 Patient Information
            </div>
            <div class="section-subtitle">
                Enter basic patient details
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    patient_name = st.text_input(
        "Patient Name", placeholder="Enter patient name"
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type",
        ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [False, True],
        format_func=lambda x: "Yes" if x else "No",
    )
    restecg = st.selectbox(
        "Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"]
    )


# ============================================================
# HEART PARAMETERS
# ============================================================

with right_column:
    st.markdown(
        textwrap.dedent("""
        <div class="section-header">
            <div class="section-title">
                ❤️ Heart Parameters
            </div>
            <div class="section-subtitle">
                Enter cardiovascular measurements
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    trestbps = st.number_input(
        "Resting Blood Pressure", min_value=50, max_value=250, value=120, step=1
    )
    chol = st.number_input(
        "Cholesterol", min_value=50, max_value=700, value=200, step=1
    )
    thalch = st.number_input(
        "Maximum Heart Rate", min_value=50, max_value=250, value=150, step=1
    )
    exang = st.selectbox(
        "Exercise Induced Angina",
        [False, True],
        format_func=lambda x: "Yes" if x else "No",
    )
    oldpeak = st.number_input(
        "Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1
    )
    slope = st.selectbox(
        "ST Segment Slope", ["downsloping", "flat", "upsloping"]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button("🔍 CHECK HEART RISK")


# ============================================================
# PREDICTION
# ============================================================

if predict_button:
    if patient_name.strip() == "":
        st.error("⚠️ Please enter the patient name.")
        st.stop()

    data = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalch,
        "oldpeak": oldpeak,
        "sex_Male": 1 if sex == "Male" else 0,
        "cp_atypical angina": 1 if cp == "atypical angina" else 0,
        "cp_non-anginal": 1 if cp == "non-anginal" else 0,
        "cp_typical angina": 1 if cp == "typical angina" else 0,
        "fbs_True": 1 if fbs else 0,
        "restecg_normal": 1 if restecg == "normal" else 0,
        "restecg_st-t abnormality": 1 if restecg == "st-t abnormality" else 0,
        "exang_True": 1 if exang else 0,
        "slope_flat": 1 if slope == "flat" else 0,
        "slope_upsloping": 1 if slope == "upsloping" else 0,
    }

    input_data = pd.DataFrame([data])

    missing_features = [
        feature for feature in feature_columns if feature not in input_data.columns
    ]

    if missing_features:
        st.error("The following features are missing from the input:")
        st.write(missing_features)
        st.stop()

    input_data = input_data[feature_columns]

    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if 1 in classes:
                positive_index = classes.index(1)
                probability = probabilities[positive_index]
            else:
                probability = probabilities[-1]
        else:
            probability = probabilities[-1]

        probability_percent = probability * 100

    except Exception as error:
        st.error("Prediction failed.")
        st.code(str(error))
        st.stop()

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        result_text = "Higher Likelihood of Heart Disease"
        st.markdown(
            f"""
            <div class="result-box result-risk">
                <div class="result-title">
                    ⚠️ Higher Likelihood of Heart Disease
                </div>
                <div class="percentage">
                    {probability_percent:.2f}%
                </div>
                <div class="result-description">
                    Estimated probability of the positive class
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        result_text = "Lower Likelihood of Heart Disease"
        st.markdown(
            f"""
            <div class="result-box result-safe">
                <div class="result-title">
                    🟢 Lower Likelihood of Heart Disease
                </div>
                <div class="percentage">
                    {probability_percent:.2f}%
                </div>
                <div class="result-description">
                    Estimated probability of the positive class
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📈 Heart Disease Probability")
    st.progress(min(max(float(probability), 0.0), 1.0))

    st.markdown(
        f"""
        <div style="
            text-align: center;
            font-size: 30px;
            font-weight: 800;
            color: #075985;
            margin: 15px;
        ">
            {probability_percent:.2f}%
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Patient Summary")

    summary = pd.DataFrame(
        {
            "Parameter": [
                "Patient Name",
                "Age",
                "Sex",
                "Chest Pain",
                "Fasting Blood Sugar > 120",
                "Resting ECG",
                "Blood Pressure",
                "Cholesterol",
                "Maximum Heart Rate",
                "Exercise Angina",
                "Oldpeak",
                "Slope",
                "Result",
                "Heart Disease Probability",
            ],
            "Value": [
                patient_name,
                age,
                sex,
                cp,
                "Yes" if fbs else "No",
                restecg,
                trestbps,
                chol,
                thalch,
                "Yes" if exang else "No",
                oldpeak,
                slope,
                result_text,
                f"{probability_percent:.2f}%",
            ],
        }
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("### 💡 Interpretation")

    if prediction == 1:
        st.warning(
            f"""
            The Random Forest model predicts a
            **higher likelihood of the positive class**.

            Predicted probability:
            **{probability_percent:.2f}%**
            """
        )
    else:
        st.success(
            f"""
            The Random Forest model predicts a
            **lower likelihood of the positive class**.

            Predicted probability:
            **{probability_percent:.2f}%**
            """
        )

    st.warning(
        """
        ⚠️ **Important:** This application is developed for
        academic and research purposes only.

        The prediction generated by this machine-learning model
        is **not a medical diagnosis** and should not replace
        professional medical advice.
        """
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.markdown(
    textwrap.dedent("""
    <div class="info-card">
        <div class="info-title">
            🧠 About This Project
        </div>
        <p class="info-text">
            The <b>Heart Health Predictor</b> is an academic
            machine-learning project designed to demonstrate how
            healthcare-related patient data can be used for
            predictive analysis.
        </p>
        <p class="info-text">
            The system uses a
            <b>Random Forest Classification Model</b>
            trained on heart disease-related data.
        </p>
        <p class="info-text">
            The model uses parameters such as age,
            blood pressure, cholesterol, maximum heart rate,
            chest pain type and exercise-induced angina.
        </p>
        <p class="info-text">
            The project workflow includes data preprocessing,
            missing-value handling, categorical encoding,
            model training, model evaluation, ROC-AUC analysis,
            feature importance and SHAP-based interpretation.
        </p>
        <p class="info-text">
            The purpose of this project is to demonstrate the
            application of <b>Machine Learning in healthcare
            data analysis</b>.
        </p>
    </div>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# DEVELOPER
# ============================================================

if profile_image:
    st.markdown(
        f"""
        <div class="developer-card">
            <img
                class="developer-image"
                src="data:image/jpeg;base64,{profile_image}"
            >
            <div class="developer-name">
                Developed by Subhanshu Yagik
            </div>
            <p class="developer-role">
                B.Tech Data Science Student
            </p>
            <p class="developer-role">
                Academic Machine Learning Project
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="developer-card">
            <div class="developer-name">
                Developed by Subhanshu Yagik
            </div>
            <p class="developer-role">
                B.Tech Data Science Student
            </p>
            <p class="developer-role">
                Academic Machine Learning Project
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    textwrap.dedent("""
    <div class="footer-box">
        <div class="footer-title">
            ❤️ Heart Health Predictor
        </div>
        <div>
            Random Forest Machine Learning Model
        </div>
        <div>
            Developed by <b>Subhanshu Yagik</b>
        </div>
        <div>
            Academic & Research Project
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

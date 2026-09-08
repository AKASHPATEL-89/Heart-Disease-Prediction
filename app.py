
import streamlit as st
import pandas as pd
import joblib

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =================================================
# CUSTOM CSS
# =================================================
st.markdown("""
<style>

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #eef2ff, #f8fafc);
    }

    /* Remove Streamlit default top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* HERO SECTION */
    .hero {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        padding: 45px 30px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0px 12px 30px rgba(37, 99, 235, 0.25);
    }

    .hero h1 {
        color: white !important;
        font-size: 48px !important;
        font-weight: 800 !important;
        margin-bottom: 10px !important;
    }

    .hero p {
        color: #f1f5f9 !important;
        font-size: 20px !important;
        margin: 0 !important;
    }

    /* Section Headings */
    .section-title {
        color: #1e293b !important;
        font-size: 28px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Input Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 60px;
        border-radius: 14px;
        border: none;
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white !important;
        font-size: 20px;
        font-weight: 700;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.02);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 25px;
        color: #475569 !important;
        font-size: 15px;
    }

</style>
""", unsafe_allow_html=True)


# =================================================
# LOAD MODEL
# =================================================
@st.cache_resource
def load_model():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")

    return model, scaler, expected_columns


model, scaler, expected_columns = load_model()


# =================================================
# HERO HEADER
# =================================================
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Prediction</h1>
    <p>AI-Powered Health Risk Analysis System</p>
</div>
""", unsafe_allow_html=True)


# =================================================
# INFORMATION
# =================================================
st.info(
    "🩺 Enter the health information below and click Predict to get a machine-learning-based risk assessment."
)


# =================================================
# INPUT FORM
# =================================================
st.markdown(
    '<div class="section-title">📝 Patient Health Information</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("👤 Personal Information")

        age = st.slider("Age", 18, 100, 40)

        sex = st.selectbox(
            "Sex",
            ["M", "F"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            80,
            200,
            120
        )

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            100,
            600,
            200
        )


    with col2:

        st.subheader("❤️ Heart & Exercise Information")

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1]
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        max_hr = st.slider(
            "Maximum Heart Rate",
            60,
            220,
            150
        )

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["Y", "N"]
        )

        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            0.0,
            6.0,
            1.0,
            0.1
        )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )


    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🔍 Analyze Heart Disease Risk",
        use_container_width=True
    )


# =================================================
# PREDICTION
# =================================================
if submitted:

    with st.spinner("🤖 AI is analyzing the health information..."):

        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,

            "Sex_" + sex: 1,
            "ChestPainType_" + chest_pain: 1,
            "RestingECG_" + resting_ecg: 1,
            "ExerciseAngina_" + exercise_angina: 1,
            "ST_Slope_" + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        # Add missing columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Correct column order
        input_df = input_df[expected_columns]

        # Scale data
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_input)[0]


    # =================================================
    # RESULT
    # =================================================
    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:

        st.error(
            "⚠️ **Higher Predicted Risk**\n\n"
            "The machine learning model detected patterns associated with a higher predicted risk."
        )

    else:

        st.success(
            "✅ **Lower Predicted Risk**\n\n"
            "The machine learning model detected a lower predicted risk based on the entered information."
        )

    st.warning(
        "⚕️ **Important:** This application is an educational machine-learning project and does not provide a medical diagnosis."
    )


# =================================================
# FOOTER
# =================================================
st.markdown("""
<div class="footer">
    ❤️ <b>Heart Disease Prediction System</b><br>
    Developed by <b>Akash Singh</b> • Machine Learning • Streamlit
</div>
""", unsafe_allow_html=True)


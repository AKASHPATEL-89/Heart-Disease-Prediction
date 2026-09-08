
import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

    .main {
        background-color: #f5f7fb;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
    }

    div.stButton > button {
        width: 100%;
        height: 55px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():

    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")

    return model, scaler, expected_columns


model, scaler, expected_columns = load_model()


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Heart Health Risk Prediction System</div>',
    unsafe_allow_html=True
)


st.markdown("""
<div class="info-box">
🩺 Enter the patient's health details below to analyze the potential risk of heart disease.
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------

st.subheader("📝 Patient Health Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.slider(
        "🎂 Age",
        min_value=18,
        max_value=100,
        value=40
    )

    sex = st.selectbox(
        "👤 Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "💢 Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "🩸 Resting Blood Pressure",
        min_value=80,
        max_value=200,
        value=120
    )


with col2:

    cholesterol = st.number_input(
        "🧪 Cholesterol",
        min_value=100,
        max_value=600,
        value=200
    )

    fasting_bs = st.selectbox(
        "🍬 Fasting Blood Sugar > 120",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "📈 Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "❤️ Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )


with col3:

    exercise_angina = st.selectbox(
        "🏃 Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "📉 Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "📊 ST Slope",
        ["Up", "Flat", "Down"]
    )


# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------

st.markdown("---")

if st.button("🔍 Predict Heart Disease Risk"):

    with st.spinner("🤖 AI is analyzing the health information..."):

        # Create raw input
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Create DataFrame
        input_df = pd.DataFrame([raw_input])

        # Add missing columns
        for col in expected_columns:

            if col not in input_df.columns:
                input_df[col] = 0

        # Arrange columns
        input_df = input_df[expected_columns]

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_input)[0]


    # -------------------------------------------------
    # RESULT
    # -------------------------------------------------

    st.markdown("---")
    st.subheader("📋 Prediction Result")

    if prediction == 1:

        st.error("""
        ⚠️ **Higher Predicted Risk**

        The model indicates a higher predicted likelihood based on the entered data.

        **Please consult a qualified healthcare professional for proper medical advice.**
        """)

    else:

        st.success("""
        ✅ **Lower Predicted Risk**

        The model indicates a lower predicted likelihood based on the entered data.

        **This result is not a medical diagnosis. Please consult a healthcare professional for concerns or symptoms.**
        """)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <center>
        ❤️ <b>Heart Disease Prediction System</b><br>
        Developed by <b>Akash Singh</b> | Powered by Machine Learning & Streamlit
    </center>
    """,
    unsafe_allow_html=True
)



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
st.title("❤️ Heart Disease Prediction")
st.caption("Machine Learning Based Heart Health Risk Prediction System")

st.divider()

st.info(
    "Enter the patient health information below to get a prediction."
)


# -------------------------------------------------
# INPUT FORM
# -------------------------------------------------
st.subheader("Patient Information")

with st.form("heart_prediction_form"):

    col1, col2 = st.columns(2)

    # LEFT COLUMN
    with col1:

        age = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=40
        )

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
            min_value=80,
            max_value=200,
            value=120
        )

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=600,
            value=200
        )


    # RIGHT COLUMN
    with col2:

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
            min_value=60,
            max_value=220,
            value=150
        )

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["Y", "N"]
        )

        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )


    st.divider()

    submitted = st.form_submit_button(
        "Predict Heart Disease Risk",
        use_container_width=True
    )


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if submitted:

    with st.spinner("Analyzing patient information..."):

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

        # Create DataFrame
        input_df = pd.DataFrame([raw_input])

        # Add missing columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Arrange columns correctly
        input_df = input_df[expected_columns]

        # Scale input data
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_input)[0]


    # -------------------------------------------------
    # RESULT
    # -------------------------------------------------
    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Higher predicted risk of heart disease.")
    else:
        st.success("✅ Lower predicted risk of heart disease.")

    st.caption(
        "Disclaimer: This application is an educational Machine Learning project "
        "and should not be used as a substitute for professional medical advice."
    )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()

st.caption(
    "Developed by Akash Singh | Heart Disease Prediction | Streamlit & Machine Learning"
)


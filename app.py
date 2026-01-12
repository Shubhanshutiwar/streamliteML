import streamlit as st
import joblib
import numpy as np

model = joblib.load("student_pass_model3.pkl")
st.title("Student Pass/Fail Prediction System")

study_hours = st.number_input(
    "Study Hours per Week", min_value=0.0, max_value=40.0, value=10.0
)

attendance = st.number_input(
    "Attendance Percentage", min_value=0.0, max_value=100.0, value=75.0
)

marks = st.number_input(
    "Marks", min_value=0.0, max_value=100.0, value=50.0
)

if st.button("Predict Result"):
    input_data = np.array([[study_hours, attendance, marks]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("Prediction: PASS")
    else:
        st.error("Prediction: FAIL")
          
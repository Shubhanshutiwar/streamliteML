import streamlit as st
import joblib
import numpy as np
import os

# 1. Added a check to ensure the model file exists before loading
model_path = "student_pass_model3.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Model file '{model_path}' not found. Please ensure it is in the same directory.")
    st.stop() 

st.title("Student Pass/Fail Prediction System")

# 2. Organized layout using columns for a cleaner look
col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "Study Hours per Week", min_value=0.0, max_value=168.0, value=10.0
    )
    attendance = st.number_input(
        "Attendance Percentage", min_value=0.0, max_value=100.0, value=75.0
    )

with col2:
    marks = st.number_input(
        "Current Marks", min_value=0.0, max_value=100.0, value=50.0
    )

# 3. Prediction Logic
if st.button("Predict Result", type="primary"):
    # Ensure features are in the exact order the model was trained on
    input_data = np.array([[study_hours, attendance, marks]])
    
    try:
        prediction = model.predict(input_data)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.success("### Prediction: **PASS**")
            st.balloons()
        else:
            st.error("### Prediction: **FAIL**")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


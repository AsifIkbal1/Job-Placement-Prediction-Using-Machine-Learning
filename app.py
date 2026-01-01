import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Job Placement Prediction Web App")
st.write("Enter feature values manually to predict status using SVM model.")

# 1️⃣ Load pickle model (fixed path)
pickle_path = '/Users/apple/Downloads/Project/Job-Placement-Prediction-Using-Machine-Learning/research/svm_model.pkl'

try:
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        model = data['model']
        le_dict = data['encoders']
    else:
        model = data
        le_dict = {}

    st.success("Pickle model loaded successfully!")

except FileNotFoundError:
    st.error(f"Pickle file not found at {pickle_path}.")
except Exception as e:
    st.error(f"Error loading pickle: {e}")

# 2️⃣ Manual input for all columns
st.subheader("Enter Feature Values:")

input_data = {}

# List of all columns
features = ['gender', 'ssc_percentage', 'ssc_board', 'hsc_percentage', 'hsc_board',
            'hsc_subject', 'degree_percentage', 'undergrad_degree', 'work_experience',
            'emp_test_percentage', 'specialisation', 'mba_percent', 'years_experience',
            'skills_match_percent', 'num_certifications', 'internship_completed',
            'interview_score', 'company_tier', 'job_competition_level']

# Create input widgets
for col in features:
    if col in le_dict:  # categorical
        input_data[col] = st.selectbox(col, le_dict[col].classes_)
    else:  # numerical
        input_data[col] = st.number_input(col, value=0.0)

# 3️⃣ Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical columns
for col, le in le_dict.items():
    input_df[col] = le.transform(input_df[col])

# Ensure contiguous array
input_df = np.ascontiguousarray(input_df)

# 4️⃣ Prediction
if st.button("Predict Status"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Status: {prediction}")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
            st.info(f"Probability of Positive Class: {probability:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Job Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Job Placement Prediction System")
st.markdown(
    """
    This web application predicts whether a student will be **Placed or Not Placed**
    based on academic performance, skills, and experience using a trained **SVM model**.
    """
)

# Load model
pickle_path = '/Users/apple/Downloads/Project/Job-Placement-Prediction-Using-Machine-Learning/research/svm_model.pkl'

try:
    with open(pickle_path, 'rb') as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

st.divider()

# =========================
# INPUT SECTION
# =========================
st.subheader("📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📘 Academic Information")
    ssc_percentage = st.number_input("SSC Percentage", 0.0, 100.0)
    ssc_board = st.selectbox("SSC Board", ["Others", "Central"])
    hsc_percentage = st.number_input("HSC Percentage", 0.0, 100.0)
    hsc_board = st.selectbox("HSC Board", ["Others", "Central"])
    hsc_subject = st.selectbox("HSC Subject", ["Arts", "Commerce", "Science"])
    degree_percentage = st.number_input("Degree Percentage", 0.0, 100.0)
    undergrad_degree = st.selectbox("Undergraduate Degree", ["Arts", "Commerce", "Science"])
    mba_percent = st.number_input("MBA Percentage", 0.0, 100.0)

with col2:
    st.markdown("### 💼 Skills & Experience")
    gender = st.selectbox("Gender", ["Female", "Male"])
    work_experience = st.selectbox("Work Experience", ["No", "Yes"])
    emp_test_percentage = st.number_input("Employability Test Score", 0.0, 100.0)
    specialisation = st.selectbox("Specialisation", ["HR", "Marketing"])
    years_experience = st.number_input("Years of Experience", 0, 20)
    skills_match_percent = st.number_input("Skills Match Percentage", 0.0, 100.0)
    num_certifications = st.number_input("Number of Certifications", 0, 10)
    internship_completed = st.selectbox("Internship Completed", ["No", "Yes"])
    interview_score = st.number_input("Interview Score", 0.0, 10.0)
    company_tier = st.selectbox("Company Tier", ["Startup", "Mid", "MNC"])
    job_competition_level = st.selectbox("Job Competition Level", ["Low", "Medium", "High"])

st.divider()

# =========================
# ENCODING (MANUAL)
# =========================
input_data = {
    'gender': 1 if gender == "Male" else 0,
    'ssc_percentage': ssc_percentage,
    'ssc_board': 1 if ssc_board == "Central" else 0,
    'hsc_percentage': hsc_percentage,
    'hsc_board': 1 if hsc_board == "Central" else 0,
    'hsc_subject': {"Arts": 0, "Commerce": 1, "Science": 2}[hsc_subject],
    'degree_percentage': degree_percentage,
    'undergrad_degree': {"Arts": 0, "Commerce": 1, "Science": 2}[undergrad_degree],
    'work_experience': 1 if work_experience == "Yes" else 0,
    'emp_test_percentage': emp_test_percentage,
    'specialisation': 0 if specialisation == "HR" else 1,
    'mba_percent': mba_percent,
    'years_experience': years_experience,
    'skills_match_percent': skills_match_percent,
    'num_certifications': num_certifications,
    'internship_completed': 1 if internship_completed == "Yes" else 0,
    'interview_score': interview_score,
    'company_tier': {"Startup": 0, "Mid": 1, "MNC": 2}[company_tier],
    'job_competition_level': {"Low": 0, "Medium": 1, "High": 2}[job_competition_level]
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])
input_df = np.ascontiguousarray(input_df)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Placement Status", use_container_width=True):
    try:
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("🎉 **Result: Placed**")
        else:
            st.error("❌ **Result: Not Placed**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

# 1️⃣ Load your dataset
# df = pd.read_csv('your_dataset.csv')  # Uncomment if reading CSV

# 2️⃣ Separate features and target
X = df[['gender', 'ssc_percentage', 'ssc_board', 'hsc_percentage', 'hsc_board',
       'hsc_subject', 'degree_percentage', 'undergrad_degree', 'work_experience',
       'emp_test_percentage', 'specialisation', 'mba_percent', 'years_experience',
       'skills_match_percent', 'num_certifications', 'internship_completed',
       'interview_score', 'company_tier', 'job_competition_level']]

y = df['status']

# 3️⃣ Label encode categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns
le_dict = {}  # Save encoders to use later in UI

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# 4️⃣ Train SVM
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X, y)

# 5️⃣ Save SVM model + LabelEncoders as a pickle
with open('svm_model.pkl', 'wb') as f:
    pickle.dump({'model': svm_model, 'encoders': le_dict}, f)

print("SVM model with encoders saved as 'svm_model.pkl'")

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load dataset
data = pd.read_csv("student_exam_data_new.csv")

# 2️⃣ Select only FEATURE columns (NOT target column)
# Example:
# If your dataset columns are:
# marks, attendance, study_hours, result

X = data[['Study Hours', 'Previous Exam Score']]   # change to your actual feature names

# 3️⃣ Create scaler
scaler = StandardScaler()

# 4️⃣ Fit scaler on dataset
scaler.fit(X)

# 5️⃣ Save scaler as scaler.pkl
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ scaler.pkl created successfully!")

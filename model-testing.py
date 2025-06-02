# predict_task.py

import joblib
import pandas as pd
import scipy.sparse

# Load the saved model, encoder, and scaler
model = joblib.load('trained_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Define a new task example
task_example = {
    "Required Skills": ["Python", "Security", "API Development"],
    "Estimated Effort": 5,
    "Priority": "High",
    "Dependencies": ["Database Setup", "Frontend Development"],
    "Current Workload": 6,
    "Experience Level": "Senior",
    "Days_Until_Deadline": 10
}

# Convert task data to DataFrame
task_df = pd.DataFrame([task_example])

# Feature engineering (match the training format)
task_df["Dependencies_Count"] = task_df["Dependencies"].apply(len)

# Encode categorical features
task_df_categorical = encoder.transform(task_df[["Required Skills", "Priority", "Experience Level"]])

# Scale numerical features
task_df_numerical = scaler.transform(task_df[["Estimated Effort", "Team Member Workload", "Dependencies_Count", "Days_Until_Deadline"]])

# Concatenate features into final feature matrix
task_example_processed = scipy.sparse.hstack([task_df_categorical, task_df_numerical])

# Predict the team member for the new task
predicted_team_member = model.predict(task_example_processed)

print("Predicted Team Member for Task:", predicted_team_member[0])

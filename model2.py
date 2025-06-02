# model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
import joblib
import scipy.sparse

# Load and preprocess the data
df = pd.read_csv('C:\large_task_distribution_dataset.csv')

# Feature engineering (example)
df['Dependencies_Count'] = df['Task Dependencies'].apply(lambda x: len(eval(x)) if pd.notna(x) else 0)
df['Deadline'] = pd.to_datetime(df['Deadline'])
df['Creation Date'] = pd.to_datetime(df['Creation Date'])
df['Days_Until_Deadline'] = (df['Deadline'] - df['Creation Date']).dt.days

# Features and target
X = df[['Required Skills', 'Estimated Effort', 'Priority', 'Dependencies_Count', 'Team Member Workload', 'Experience Level', 'Days_Until_Deadline']]
y = df['Assigned Team Member ID']

# Ensure y is 1-dimensional
y = y.values.ravel()  # This will flatten y to shape (10000,)

# MultiLabelBinarizer for 'Required Skills'
mlb = MultiLabelBinarizer()
X_skills = mlb.fit_transform(X['Required Skills'].apply(eval))  # Convert stringified lists to lists

# OneHotEncoder for other categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_other_categorical = encoder.fit_transform(X[['Priority', 'Experience Level']])

# Scale numerical features
numerical_features = ['Estimated Effort', 'Team Member Workload', 'Dependencies_Count', 'Days_Until_Deadline']
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[numerical_features])

# Combine all preprocessed features
X_processed = scipy.sparse.hstack([X_skills, X_other_categorical, X_numerical])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save model, encoders, and scaler
joblib.dump(model, 'trained_model.pkl')
joblib.dump(mlb, 'mlb.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model, encoders, and scaler saved successfully!")

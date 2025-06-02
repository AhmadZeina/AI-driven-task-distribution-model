# 1. Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler , MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
import joblib 

#made some changes from ''optimize'' branch 

value = True









# 2. Load the Dataset
# Replace 'path_to_dataset.csv' with the actual path to your dataset file
df = pd.read_csv('C:\large_task_distribution_dataset.csv')

# 3. Feature Engineering
# We’ll create new features and encode categorical ones for better model performance.

# Feature: Count of dependencies
df['Dependencies_Count'] = df['Task Dependencies'].apply(lambda x: len(eval(x)) if pd.notna(x) else 0)

# Feature: Deadline Days (days until deadline from creation date)
df['Deadline'] = pd.to_datetime(df['Deadline'])
df['Creation Date'] = pd.to_datetime(df['Creation Date'])
df['Days_Until_Deadline'] = (df['Deadline'] - df['Creation Date']).dt.days

# Encode Required Skills and Priority as categorical variables.
# We will also encode Experience Level and Priority, which are categorical.

# 4. Selecting and Preprocessing Features and Labels
# Define your features and target
X = df[['Required Skills', 'Estimated Effort', 'Priority', 'Dependencies_Count', 'Team Member Workload', 'Experience Level', 'Days_Until_Deadline']]
y = df['Assigned Team Member ID']


# Initialize MultiLabelBinarizer for 'Required Skills'
mlb = MultiLabelBinarizer()
X_skills = mlb.fit_transform(X['Required Skills'].apply(eval))  # Convert stringified lists back to lists



# Initialize OneHotEncoder for categorical features and StandardScaler for continuous features
categorical_features = ['Required Skills', 'Priority', 'Experience Level']
numerical_features = ['Estimated Effort', 'Team Member Workload', 'Dependencies_Count', 'Days_Until_Deadline']

# Encode categorical features
encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(X[categorical_features])

# Scale numerical features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[numerical_features])

# Concatenate categorical and numerical features into the final feature matrix
import scipy.sparse
X_processed = scipy.sparse.hstack([X_categorical, X_numerical])

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 6. Model Selection and Training
# Define and train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 7. Predictions and Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model with accuracy and a detailed classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save model, encoder, and scaler to disk
joblib.dump(model, 'trained_model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Model, encoder, and scaler saved successfully!")
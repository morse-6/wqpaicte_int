import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import os

# Load data
df = pd.read_csv("data.csv", sep=';')

# Drop non-numeric columns
df = df.drop(columns=['id', 'date'])

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split for training and testing (optional)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Create folder for model artifacts
os.makedirs('model', exist_ok=True)

# Save model and scaler
joblib.dump(model, 'model/svm.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Model and scaler saved to 'model/' folder successfully.")

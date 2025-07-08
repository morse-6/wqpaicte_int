import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load your dataset (no target column)
df = pd.read_csv("data.csv", sep=';')

# Drop non-feature columns
df_features = df.drop(columns=['id', 'date'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# Save the scaler for future use
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')

print("Scaler saved. No model trained because target is missing.")

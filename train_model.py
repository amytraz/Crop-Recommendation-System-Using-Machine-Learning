import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your CSV
df = pd.read_csv("Crop_recommendation.csv")

# Features and target
X = df.drop('label', axis=1)
y = df['label']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply MinMaxScaler and StandardScaler
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)

standard_scaler = StandardScaler()
X_scaled = standard_scaler.fit_transform(X_minmax)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scalers
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("minmaxscaler.pkl", "wb") as f:
    pickle.dump(minmax_scaler, f)

with open("standscaler.pkl", "wb") as f:
    pickle.dump(standard_scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model and scalers saved successfully.")

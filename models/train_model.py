import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Paths
DATA_PATH = "Crop_recommendation.csv"
SAVE_DIR = "."

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Feature Engineering
df["npk_sum"] = df["N"] + df["P"] + df["K"]
df["npk_mean"] = (df["N"] + df["P"] + df["K"]) / 3
df["temp_rain_interaction"] = df["temperature"] * df["rainfall"]

# Encode target
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Save label encoder
with open(os.path.join(SAVE_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# Split features and target
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# Dictionary to store accuracies
accuracies = {}

# Train, evaluate, and save models
print("\n🔧 Training Models...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc * 100

    # Save model
    with open(os.path.join(SAVE_DIR, f"{name.lower().replace(' ', '_')}.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"✅ {name} trained and saved.")

# Print accuracy summary
print("\n📊 Model Accuracy Summary:\n" + "-" * 30)
for name, acc in accuracies.items():
    print(f"{name:15}: {acc:.2f}%")
print("-" * 30)
print("\n📁 All models and encoder saved successfully.")

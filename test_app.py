import pickle
import pandas as pd
from collections import Counter

# Utility to load models
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Load trained models
models = {
    "Naive Bayes": load_model("models/naive_bayes.pkl"),
    "Random Forest": load_model("models/random_forest.pkl"),
    "XGBoost": load_model("models/xgboost.pkl"),
    "LightGBM": load_model("models/lightgbm.pkl")
}

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Prediction function with engineered features
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = {
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    }

    input_df = pd.DataFrame(input_data)

    # Feature Engineering
    input_df["npk_sum"] = input_df["N"] + input_df["P"] + input_df["K"]
    input_df["npk_mean"] = (input_df["N"] + input_df["P"] + input_df["K"]) / 3
    input_df["temp_rain_interaction"] = input_df["temperature"] * input_df["rainfall"]

    predictions = []
    failed_models = []

    for name, model in models.items():
        try:
            pred = model.predict(input_df)[0]
            predictions.append(pred)
        except Exception as e:
            print(f"⚠️ Error predicting with {name}: {e}")
            predictions.append(None)
            failed_models.append(name)

    # Majority voting (excluding failed predictions)
    filtered_preds = [p for p in predictions if p is not None]
    final_encoded = Counter(filtered_preds).most_common(1)[0][0]
    final_prediction = label_encoder.inverse_transform([final_encoded])[0]

    # Decode all predictions (handling None)
    all_preds = [
        label_encoder.inverse_transform([p])[0] if p is not None else "❌ Error"
        for p in predictions
    ]

    return final_prediction, all_preds

# Test cases
test_cases = [
    {
        "name": "Test Case 1 🌾",
        "input": {
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.5, "humidity": 82.0,
            "ph": 6.5, "rainfall": 200.0
        }
    },
    {
        "name": "Test Case 2",
        "input": {
            "N": 25, "P": 45, "K": 22,
            "temperature": 26.3, "humidity": 85.0,
            "ph": 6.2, "rainfall": 100.0
        }
    },
    {
        "name": "Test Case 3",
        "input": {
            "N": 105, "P": 35, "K": 40,
            "temperature": 23.0, "humidity": 65.0,
            "ph": 6.8, "rainfall": 190.0
        }
    },
    {
        "name": "Test Case 4 🌾",
        "input": {
            "N": 45, "P": 50, "K": 44,
            "temperature": 27.5, "humidity": 90.0,
            "ph": 6.1, "rainfall": 120.0
        }
    },
    {
        "name": "Test Case 5 🧪",
        "input": {
            "N": 80, "P": 40, "K": 35,
            "temperature": 22.8, "humidity": 70.0,
            "ph": 6.5, "rainfall": 150.0
        }
    }
]

# Run test predictions
print("🌿 Crop Recommendation Test Results:")
print("---------------------------------------------")
for case in test_cases:
    predicted, all_preds = predict_crop(**case["input"])
    print(f"{case['name']}:")
    print(f"  ✅ Final Prediction 🌾: {predicted}")
    print(f"  🔍 Model-wise Predictions: {dict(zip(models.keys(), all_preds))}")
    print()

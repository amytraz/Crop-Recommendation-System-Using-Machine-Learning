from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
from collections import Counter

# Flask app setup
app = Flask(__name__)

# Load models
model_paths = {
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "LightGBM": "models/lightgbm.pkl"
}

models = {name: pickle.load(open(path, "rb")) for name, path in model_paths.items()}

# Load label encoder
le = pickle.load(open('models/label_encoder.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html", result=None, data={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = {
            'N': float(request.form['Nitrogen']),
            'P': float(request.form['Phosporus']),
            'K': float(request.form['Potassium']),
            'temperature': float(request.form['Temperature']),
            'humidity': float(request.form['Humidity']),
            'ph': float(request.form['Ph']),
            'rainfall': float(request.form['Rainfall'])
        }

        # Create DataFrame
        input_df = pd.DataFrame([form_data])

        # Feature Engineering
        input_df["npk_sum"] = input_df["N"] + input_df["P"] + input_df["K"]
        input_df["npk_mean"] = (input_df["N"] + input_df["P"] + input_df["K"]) / 3
        input_df["temp_rain_interaction"] = input_df["temperature"] * input_df["rainfall"]

        # Predict with all models
        predictions = [model.predict(input_df)[0] for model in models.values()]
        final_encoded = Counter(predictions).most_common(1)[0][0]
        final_prediction = le.inverse_transform([final_encoded])[0]

        # Decode individual predictions
        model_wise_preds = {name: le.inverse_transform([pred])[0] for name, pred in zip(models.keys(), predictions)}

        return render_template("index.html", result=final_prediction, data=form_data, model_preds=model_wise_preds)

    except Exception as e:
        return render_template("index.html", result=f"🌾 Error: {str(e)} 🌿", data=request.form)

if __name__ == "__main__":
    app.run(debug=True)

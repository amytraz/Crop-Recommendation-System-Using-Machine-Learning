from flask import Flask, request, render_template
import numpy as np
import pickle

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", result=None, data={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        form_data = {
            'Nitrogen': float(request.form['Nitrogen']),
            'Phosporus': float(request.form['Phosporus']),
            'Potassium': float(request.form['Potassium']),
            'Temperature': float(request.form['Temperature']),
            'Humidity': float(request.form['Humidity']),
            'Ph': float(request.form['Ph']),
            'Rainfall': float(request.form['Rainfall'])
        }

        # Prepare and scale features
        features = np.array([[form_data['Nitrogen'], form_data['Phosporus'], form_data['Potassium'],
                              form_data['Temperature'], form_data['Humidity'], form_data['Ph'], form_data['Rainfall']]])
        features_minmax = ms.transform(features)
        features_scaled = sc.transform(features_minmax)

        # Predict
        pred_class = model.predict(features_scaled)[0]
        crop_name = le.inverse_transform([pred_class])[0]

        return render_template("index.html", result=crop_name, data=form_data)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}", data=request.form)

if __name__ == "__main__":
    app.run(debug=True)

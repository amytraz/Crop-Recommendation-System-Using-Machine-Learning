🌾 Crop Recommendation System 🌿
This is a web-based machine learning application that recommends the most suitable crop to grow based on environmental conditions and soil parameters. It uses a trained machine learning model to predict the best crop using user-inputted values like nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

🔍 Overview
The system uses user inputs from a web form and processes them using trained scalers and a machine learning model to provide crop suggestions. It's built with Flask for the backend and Bootstrap 5 for the frontend UI.

⚙️ Features
Input soil and weather parameters

Instantly get a recommended crop based on ML prediction

Keeps form data visible after prediction

Option to make a new prediction without refreshing

User-friendly and responsive UI

🤖 Machine Learning Model
Algorithms Tested:
✅ Random Forest Classifier (Used in final model)

🔍 Decision Tree

🔍 K-Nearest Neighbors (KNN)

🔍 Logistic Regression

🔍 Support Vector Machine (SVM)

🔍 Naive Bayes

🔍 XGBoost

After evaluating all models, the Random Forest Classifier was selected due to its high accuracy and ability to handle complex decision boundaries.

🧠 Input Parameters
Nitrogen (N) – amount of nitrogen in the soil

Phosphorus (P) – amount of phosphorus in the soil

Potassium (K) – amount of potassium in the soil

Temperature – in degrees Celsius

Humidity – in percentage (%)

pH – acidity or alkalinity of the soil

Rainfall – in mm

🚀 How It Works
User fills in the form on the webpage.

The data is scaled using MinMaxScaler and StandardScaler.

The trained RandomForestClassifier model makes a prediction.

The predicted crop label is decoded using LabelEncoder.

The result is displayed on the same page, with an option for a new prediction.

👨‍💻 Tech Stack
Frontend: HTML, CSS, Bootstrap 5

Backend: Python, Flask

ML Libraries: scikit-learn, numpy, pickle

📷 Screenshot
![Landing Page](https://github.com/user-attachments/assets/1a8b488f-4651-4d7e-aaa9-a9d3421f36e6)
![working](https://github.com/user-attachments/assets/abe101c0-be45-4211-a605-dbc46ff8fcb7)


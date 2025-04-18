
# 🌾 Crop Recommendation System using Machine Learning

This project provides a smart crop recommendation system based on environmental conditions and soil parameters using machine learning models. It is designed to help farmers and agricultural advisors make data-driven decisions about which crop is best suited for cultivation in a particular area.

## 📌 Project Overview

The system accepts seven input parameters:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature (°C)
- Humidity (%)
- pH Level
- Rainfall (mm)

Using these parameters, the application predicts the most suitable crop using a trained ensemble machine learning model based on majority voting from the following classifiers:
- Random Forest
- Naive Bayes
- XGBoost
- LightGBM

## 🧠 Key Features
- High accuracy (99%+) from ensemble learning
- Majority voting for robust crop prediction
- Web-based interface for easy access and use
- Simple and interactive UI built using Flask, HTML, and CSS
- Model-wise prediction visibility for transparency

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/crop-recommendation-ml.git
cd Crop-Recommendation-System-Using-Machine-Learning
```

### 2. Create and Activate Virtual Environment (Optional)
```bash
python -m venv env
source env/bin/activate   # On Windows use: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Models 
```bash
python train_model.py
```

### 5. Start the Flask App
```bash
python app.py
```

### 6. Open in Browser
```
http://127.0.0.1:5000/
```

## 🧪 Sample Input
| N  | P  | K  | Temp (°C) | Humidity (%) | pH  | Rainfall (mm) |
|----|----|----|------------|---------------|-----|----------------|
| 90 | 42 | 43 | 20         | 80            | 6.5 | 200            |

**Output:** `Recommended Crop: Rice`


###  User Interface

![image](https://github.com/user-attachments/assets/585d003f-70bf-4b02-8463-a352e65759a4)

###  Sample Input Data

![image](https://github.com/user-attachments/assets/5c202ab9-7d55-42bd-9db7-91ddba4171bd)

###  Crop Recommendation

![image](https://github.com/user-attachments/assets/15170272-c38f-44ec-a9fe-4fdf806c7ae5)



## 📊 Models Performance

| Model          | Accuracy |
|----------------|----------|
| Random Forest  | 99.32%   |
| Naive Bayes    | 99.09%   |
| LightGBM       | 99.09%   |
| XGBoost        | 98.64%   |

**Ensemble Voting** improves stability and handles inconsistent model outputs more reliably.


## 📁 Dataset Source
[Kaggle: Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

## 🔮 Future Improvements
- Integrate with real-time weather APIs
- Add regional language support
- Build Android/iOS mobile version
- Incorporate market price intelligence
- Enable model improvement via user feedback

## 🤝 Contributions
Contributions are welcome! Feel free to fork the repository, improve the project, and submit a pull request.

Made with ❤️ for smarter farming.

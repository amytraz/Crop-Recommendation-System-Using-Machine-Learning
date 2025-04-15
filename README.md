
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


## 🧪 Sample Input
| N  | P  | K  | Temp (°C) | Humidity (%) | pH  | Rainfall (mm) |
|----|----|----|------------|---------------|-----|----------------|
| 90 | 42 | 43 | 20         | 80            | 6.5 | 200            |

**Output:** `Recommended Crop: Rice`

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

Parkinson’s Disease Detection System
Overview

Parkinson’s Disease Detection is a web-based machine learning application designed to predict the likelihood of Parkinson’s disease using clinical and voice-related features. The system leverages supervised learning models and provides confidence-based predictions through an interactive, user-friendly interface.

The application is intended for educational, research, and decision-support purposes and demonstrates the end-to-end deployment of machine learning models in a web environment.

Key Features

Predicts Parkinson’s disease likelihood using trained ML models

Supports Support Vector Machine (SVM) and Random Forest classifiers

Accepts both manual clinical inputs and voice file uploads

Automatic feature-to-model mapping for high prediction accuracy

Displays prediction confidence using visual indicators

Interactive UI built with Bootstrap and JavaScript

Modular, extensible, and well-structured architecture

Technologies Used
Backend

Python

Flask (Web framework)

Scikit-learn (Machine learning models)

NumPy / Pandas (Data processing)

Joblib (Model persistence)

Frontend

HTML5

Bootstrap (Responsive UI)

JavaScript (ES6) (Client-side interactivity)

Machine Learning

Support Vector Machine (SVM)

Random Forest Classifier

StandardScaler for feature normalization

Dataset

Parkinson’s Disease Dataset (UCI Machine Learning Repository)

Features include vocal frequency, jitter, shimmer, and other biomedical indicators

System Architecture

User inputs clinical data and/or uploads a voice recording

JavaScript collects inputs and sends them to Flask via AJAX

Flask preprocesses inputs and aligns features with trained model schema

Selected ML model (SVM or RF) performs prediction

System returns:

Disease classification result

How to Use the System

Enter clinical feature values in the input fields

(Optional) Upload a voice recording for feature extraction

Select the prediction model (SVM or Random Forest)

Click Predict

View:

Disease prediction result

Confidence/probability visualization

Model Output Interpretation

Prediction: Indicates whether Parkinson’s disease is detected

Confidence Score: Represents the model’s estimated probability

Higher confidence suggests stronger model certainty, but does not replace medical diagnosis

Accuracy and Feature Integrity

Feature-to-model mapping is strictly enforced using stored feature names

Input order during inference exactly matches training configuration

Voice features are mapped only to relevant acoustic columns

Prevents silent accuracy degradation during prediction

Limitations

Not intended for real-world medical diagnosis

Performance depends on dataset quality and feature completeness

Voice feature extraction is simplified and for demonstration purposes

Future Enhancements

Confusion matrix and ROC-AUC visualization

Explainable AI (SHAP) integration

User authentication and data logging

Cloud deployment (Docker / AWS / Heroku)

Mobile-friendly UI extension

Disclaimer

This application is developed strictly for educational and research purposes.
It is not a substitute for professional medical diagnosis.


Probability/confidence score

Frontend visualizes the result and confidence level

# Employee-Retention-Prediction
Predict whether data scientists are likely to change jobs.
![Employee Retention Deployment](https://github.com/user-attachments/assets/92596322-7c1a-4c8f-8955-53332831a6f1)

##Deployed Link: https://employee-retention-prediction-zikkxvtyf2ysfwhps9zhtk.streamlit.app/

## Power BI Dashboard on Employee Retention Prediction Analysis

<img width="601" height="338" alt="Employee Retention Dashboard" src="https://github.com/user-attachments/assets/f74aea55-aeaf-404d-841d-96b0c051b708" />

## Table of Contents

- [Tech Stack](#tech-stack-)
- [Features](#features-)
- [Project Structure](#project-structure-)
- [References & Acknowledgments](#references--acknowledgments-)

## Tech Stack

### Frameworks / Libraries
- **Python 3.10+**
- **Streamlit** – for building interactive frontend
- **scikit-learn** – for ML model building and prediction
- **Joblib** – for loading serialized ML models
- **NumPy & Pandas** – for data manipulation and feature engineering
- **Matplotlib & Seaborn** – for visualization during model evaluation
- **Imbalanced-learn (SMOTE)** – for handling class imbalance during training

## Features

- Predicts whether an employee is likely to stay or leave based on HR data
- Uses Random Forest and XGBoost classifiers for accurate retention insights
- Handles imbalanced datasets using SMOTE for fair model training
- Includes feature importance visualization to identify top retention drivers
- Clean, interactive interface built with Streamlit for real-time prediction
- Accepts employee details as input and instantly returns prediction results
- Displays probability of attrition along with the classification
- Provides key model evaluation metrics: Accuracy, Recall, Precision, F1-Score
- Offers visual analysis: Confusion Matrix, ROC Curve, and Feature Importance chart
- Fully deployable on Streamlit Cloud or can run locally with minimal setup

## Project Structure

Employee-Retention-Prediction
- `fraud_detection_app_streamlit.py`  – Streamlit frontend + prediction logic  
- `fraud_detection.pkl`              – Trained Random Forest ML model  
- `requirements.txt`                 – Project dependencies  
- `README.md`                         – Project documentation  
- `Power BI Dashboard`                - Power BI Dashboard


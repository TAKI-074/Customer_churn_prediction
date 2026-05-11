Customer Churn Prediction API

## Overview
A complete end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn using customer behavior and service usage data.

This project covers the full ML workflow:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Cross Validation
- Hyperparameter Tuning
- Model Evaluation
- Explainable AI (SHAP)
- FastAPI Deployment

## Problem Statement
Customer churn is one of the biggest challenges for telecom companies. Acquiring new customers costs significantly more than retaining existing ones.

The goal of this project is to build a Machine Learning model capable of predicting customer churn so businesses can proactively retain high-risk customers.

## Tech Stack
### Languages
- Python
### Libraries & Frameworks
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP
- FastAPI
- Uvicorn
- Pickle

## Exploratory Data Analysis (EDA)

Performed detailed analysis to understand customer behavior and churn patterns.

### EDA Includes
- Churn distribution analysis
- Tenure distribution
- Monthly charges vs churn
- Correlation heatmap
- Internet service analysis
- Feature importance visualization

## Machine Learning Models
### Logistic Regression

Used as a baseline model for binary classification.

#### Advantages
- Fast
- Interpretable
- Strong baseline performance
### Random Forest Classifier

Used for improved nonlinear learning and feature importance extraction.

#### Advantages
- Handles complex relationships
- Reduces overfitting
- Robust performance

## Cross Validation

Implemented using:

    StratifiedKFold

This improves reliability of evaluation by preserving class distribution across folds.

## Hyperparameter Tuning

Used:

    RandomizedSearchCV

to optimize Random Forest performance.

### Parameters Tuned
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf

## Model Evaluation
### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

## Visualizations
### ROC Curve

Measures model performance across classification thresholds.

### Precision-Recall Curve

Useful for evaluating imbalanced classification problems.

### Feature Importance

Top important features identified:

- Contract type
- Tenure
- Monthly charges
- Fiber optic internet
- Payment method

## Explainable AI (SHAP)

Implemented SHAP values to explain:

- Feature impact on predictions
- Per-customer churn reasoning
- Model interpretability

This improves trust and business understanding of the model.

## FastAPI Deployment

The trained model is deployed using FastAPI.

### Features

- Real-time prediction API
- Pydantic request validation
- Swagger documentation
- Health check endpoint
- Production-ready structure

## Example API Request
      {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 850,
      
        "gender_Male": 1,
        "Partner_Yes": 1,
        "Dependents_Yes": 0,
      
        "PhoneService_Yes": 1,
      
        "MultipleLines_No_phone_service": 0,
        "MultipleLines_Yes": 1,
      
        "InternetService_Fiber_optic": 1,
        "InternetService_No": 0,
      
        "OnlineSecurity_No_internet_service": 0,
        "OnlineSecurity_Yes": 1,
      
        "OnlineBackup_No_internet_service": 0,
        "OnlineBackup_Yes": 1,
      
        "DeviceProtection_No_internet_service": 0,
        "DeviceProtection_Yes": 0,
      
        "TechSupport_No_internet_service": 0,
        "TechSupport_Yes": 0,
      
        "StreamingTV_No_internet_service": 0,
        "StreamingTV_Yes": 1,
      
        "StreamingMovies_No_internet_service": 0,
        "StreamingMovies_Yes": 1,
      
        "Contract_One_year": 0,
        "Contract_Two_year": 0,
      
        "PaperlessBilling_Yes": 1,
      
        "PaymentMethod_Credit_card_automatic": 0,
        "PaymentMethod_Electronic_check": 1,
        "PaymentMethod_Mailed_check": 0
      }

  ### Response
    {
      "churn_prediction": 1,
      "churn_probability": 0.714
    }


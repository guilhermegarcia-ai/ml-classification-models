# Bank Marketing Long Term Deposit Predictor

The Bank Marketing Long Term Deposit Predictor project aims to develop a machine learning model (classification) to predict whether a client will subscribe to a long-term deposit after a marketing campaign. By analyzing a dataset of customer information collected during campaigns, the project seeks to optimize targeting and improve conversion rates.

## Overview

Banks often run marketing campaigns to offer financial products like long-term deposits to potential clients. The campaign process involves contacting clients through various communication channels, assessing their interest, and tracking responses. Decisions on which clients to target are typically based on attributes such as age, job, marital status, education, account balance, credit history, and prior campaign results.

<p align="center">
<img src="https://github.com/user-attachments/assets/d2db287c-9f6e-492e-819a-612e653545b0" width=300 height=168>
</p>

## The Data

The dataset (https://www.kaggle.com/datasets/abdelazizsami/bank-marketing/data) contains 45,211 records detailing client attributes collected during previous marketing campaigns. It includes the following features:

- **age:** Age of the client  
- **job:** Type of job  
- **marital:** Marital status  
- **education:** Level of education  
- **default:** Has credit in default?  
- **balance:** Average annual balance in euros  
- **housing:** Has a housing loan?  
- **loan:** Has a personal loan?  
- **contact:** Communication type (cellular, telephone, unknown)  
- **day:** Last contact day of the month  
- **month:** Last contact month of the year  
- **duration:** Duration of the last contact in seconds  
- **campaign:** Number of contacts performed during this campaign  
- **pdays:** Days since last contact from a previous campaign  
- **previous:** Number of contacts performed before this campaign  
- **poutcome:** Outcome of the previous marketing campaign  
- **y:** Whether the client subscribed to a long-term deposit (target variable)

## Features

### 1. Exploratory Data Analysis Notebook (eda.ipynb)
- Performs a comprehensive analysis of data distributions, missing values, outliers, and relationships between features. Includes visualizations to uncover patterns and insights that guide feature engineering and modeling decisions.

### 2. Predictive Modeling Notebook (predictive_modeling.ipynb)
- Implements the machine learning pipeline including pre-processing, feature engineering, model training, hyperparameter tuning, and evaluation using metrics suitable for imbalanced data.

### 3. Deploy with Modularized Code for FastAPI and Streamlit
- Provides an API endpoint and a user-friendly web interface for predicting long-term deposit acceptance. Includes modularized code for domain logic, services, utilities, and pre-trained model artifacts.

## Project Structure
``` bash
bank-marketing/
├── deploy
│   ├── artifacts                 # Trained models and preprocessing artifacts
│   │   └── xgb_pipeline.pkl
│   ├── data                      # Validation data or additional datasets
│   │   └── df_validate.csv
│   ├── domain                    # Definition of domain classes and objects
│   │   ├── __init__.py
│   │   └── domain.py
│   ├── service                   # Application services (prediction logic, business rules)
│   │   ├── __init__.py
│   │   └── bank_marketing_service.py
│   └── utils                     # Utility functions for loading model, apply llm and helpers
│       ├── __init__.py
│       └── utils.py
│   ├── rest_api.py               # FastAPI endpoints for predictions
│   ├── rest_api_post_predict.http # Example POST request to test local API
│   ├── streamlit_app.py          # Streamlit app for user interaction
├── eda.ipynb                     # Exploratory Data Analysis (EDA) notebook
├── predictive_modeling.ipynb     # Predictive modeling and model evaluation notebook
└── requirements.txt              # Project dependencies (pip install -r requirements.txt)
```

## API Endpoints
- `POST /predict`: Run the prediction

## Use Examples

### 1. Running prediction for new data

## Streamlit Front-End Website
- The Streamlit app allows users to input customer data and get real-time predictions via the FastAPI endpoint, including visual explanations of feature importance.

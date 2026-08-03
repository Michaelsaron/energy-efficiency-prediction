# Energy Efficiency Prediction

An end-to-end Machine Learning regression project that predicts the **Heating Load (kWh/m²)** of residential buildings using their architectural characteristics. The project covers the complete machine learning workflow including data preprocessing, feature engineering, exploratory data analysis (EDA), model comparison, evaluation, explainability, deployment, and an interactive Streamlit web application.

---

# Problem Statement

Designing energy-efficient buildings requires accurately estimating heating demand during the planning stage. Manual estimation is time-consuming and expensive.

This project builds a complete regression pipeline that:

* Cleans and preprocesses building data
* Engineers meaningful features
* Removes multicollinearity (Surface Area removed from production pipeline)
* Trains and compares **12 regression algorithms**
* Evaluates models using multiple regression metrics
* Selects the best model using evidence rather than R² alone
* Deploys the final model with an interactive Streamlit application

**Target Variable**

Heating Load (Y1)

---

# Dataset

Dataset Source

UCI Machine Learning Repository

**Energy Efficiency Dataset**

[https://archive.ics.uci.edu/ml/datasets/Energy+efficiency](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)

| Feature                   | Description                                                           |
| ------------------------- | --------------------------------------------------------------------- |
| Relative_Compactness      | Overall compactness of the building                                   |
| Surface_Area              | Total surface area (removed from production due to multicollinearity) |
| Wall_Area                 | Exterior wall area                                                    |
| Roof_Area                 | Roof area                                                             |
| Overall_Height            | Building height                                                       |
| Orientation               | Building orientation                                                  |
| Glazing_Area              | Window/glazing ratio                                                  |
| Glazing_Area_Distribution | Distribution of glazing                                               |
| Heating_Load              | Target variable                                                       |

**Dataset Summary**

* Rows: **768**
* Original Features: **8**
* Production Features: **7**
* Missing Values: **None**
* Duplicate Records: **None**
* Target: **Heating Load**

---

# Installation

## Requirements

* Python 3.12+
* pip

---

## Setup

```bash
git clone https://github.com/Michaelsaron/energy-efficiency-prediction.git

cd energy-efficiency-prediction

pip install -r requirements.txt
```

---

# Usage

## 1. Train Models

Runs:

* preprocessing
* feature engineering
* model training
* model comparison
* model selection
* saves best model
* saves metadata

```bash
python -m src.train
```

---

## 2. Evaluate Best Model

Generates

* evaluation metrics
* feature importance
* residual analysis
* learning curves
* deployment ranking

```bash
python -m src.evaluate
```

---

## 3. Make Prediction (CLI)

```bash
python -m src.predict
```

Example

```python
sample = {
    "Relative_Compactness": 0.82,
    "Wall_Area": 294,
    "Roof_Area": 110.25,
    "Overall_Height": 7,
    "Orientation": 2,
    "Glazing_Area": 0.25,
    "Glazing_Area_Distribution": 3,
}

prediction = predict_heating_load(sample)

print(prediction)
```

---

## 4. Generate Technical Report

```bash
python -m src.report_generator
```

---

## 5. Launch Streamlit

```bash
streamlit run app/app.py
```

---

## 6. Run Tests

```bash
python -m pytest tests -v
```

---

## 7. Open Notebooks

```bash
jupyter notebook notebooks/
```

---

# Project Structure

```text
energy-efficiency-prediction/

├── app/
│   └── app.py
│
├── assets/
│   └── team/
│
├── data/
│   └── energy_efficiency_data.csv
│
├── models/
│   ├── best_model.pkl
│   ├── feature_info.json
│   ├── evaluation_metrics.json
│   └── model_metadata.json
│
├── notebooks/
│   ├── eda.ipynb
│   └── model_comparison.ipynb
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── reports/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── report_generator.py
│   ├── visualization.py
│   ├── database.py
│   ├── utils.py
│   └── auth.py
│
├── tests/
│   └── test_preprocessing.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Models Compared

The project compares **12 regression algorithms**.

| Algorithm                      |
| ------------------------------ |
| Linear Regression              |
| Ridge Regression               |
| Lasso Regression               |
| Decision Tree Regressor        |
| Random Forest Regressor        |
| Gradient Boosting Regressor    |
| Support Vector Regressor (SVR) |
| XGBoost                        |
| Extra Trees Regressor          |
| AdaBoost Regressor             |
| CatBoost                       |
| LightGBM                       |

---

# Model Evaluation

Models are evaluated using

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score
* Training Time
* Train-Test Gap
* Cross Validation Score

The final model is selected using a **weighted deployment ranking**, considering:

* RMSE
* MAE
* R²
* Training Time
* Generalization Ability (Train-Test Gap)

rather than relying only on the highest R² score.

---

# Feature Engineering

The project creates production-ready features including:

* Compactness × Height Interaction
* Relative Compactness
* Feature Validation
* Production Feature Ordering

To reduce multicollinearity,

**Surface_Area** was intentionally removed from the production pipeline because it was highly correlated with Roof Area and Wall Area.

---

# Key Findings

* Relative Compactness is the strongest predictor of Heating Load.
* Overall Height significantly affects heating demand.
* Compactness and Height interaction improves prediction accuracy.
* Surface Area introduced multicollinearity and was removed from production.
* Ensemble models consistently outperformed simple linear models.
* Cross-validation confirmed good model generalization.

---

# Outputs Generated

## Figures

* Heating Load Distribution
* Feature Distributions
* Correlation Heatmap
* Feature Relationships
* Pair Plot
* Feature Importance
* SHAP Summary Plot
* SHAP Bar Plot
* Residual Analysis
* QQ Plot
* Learning Curve
* Actual vs Predicted Plot

---

## Metrics

* Model Comparison
* Deployment Ranking
* Feature Importance
* Cross Validation Scores
* Target Correlations
* Outlier Summary
* Prediction History

---

## Reports

* Technical Markdown Report

---

# Streamlit Application

The web application includes

* Home
* Project Description
* Dataset Information
* Model Information
* Single Prediction
* Batch Prediction
* Prediction History
* Model Insights
* Model Comparison
* Team Members

---

# Bonus Features

- SHAP Explainability

-  Feature Importance Analysis

-  Learning Curves

-  Residual Analysis

-  Prediction History Database

-  User Authentication

-  Batch Prediction

-  Cross Validation

-  Deployment Ranking

-  Technical Report Generator

---

# Tech Stack

* Python 3.12
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* CatBoost
* LightGBM
* Matplotlib
* Seaborn
* SHAP
* Joblib
* SQLite
* Streamlit

---

# Future Improvements

* Cloud deployment
* REST API integration
* Automated hyperparameter tuning
* Real-time energy prediction service
* Integration with BIM (Building Information Modeling)

---

# Authors

**Group 7**

Machine Learning Regression Project

Energy Efficiency Prediction

---

# License

**All Rights Reserved**

Copyright © 2026 Michael Saron.

This project is provided for academic purposes only. No part of this repository may be copied, modified, distributed, or used without prior written permission from the author.

# Bank Churn Prediction using Hyperparameter-Tuned XGBoost

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red?logo=xgboost)](https://xgboost.readthedocs.io/)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Exploration & Cleaning](#data-exploration--cleaning)
- [Data Preprocessing](#data-preprocessing)
- [Model Selection](#model-selection)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Final Model Performance](#final-model-performance)
- [Predictions](#predictions)
- [Key Takeaways](#key-takeaways)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

---

## Project Overview
This project predicts whether a bank customer will churn (leave the bank) based on historical customer data. The main focus is **maximizing recall**, since missing potential churners is costlier than giving false alarms.

---

## Dataset
- Source: [Kaggle Playground Series S4E1](https://www.kaggle.com/competitions/playground-series-s4e1/overview)
- Files:
  - `train.csv` – 165,034 rows, 14 columns.
  - `test.csv` – Test data for prediction.
  - `sample_submission.csv` – Example submission file.

---

## Data Exploration & Cleaning
- Dropped irrelevant columns: `id`, `CustomerId`, `Surname`.
- Categorical columns: `Geography`, `Gender`, `HasCrCard`, `IsActiveMember`.
- Numerical columns: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`.
- Outliers removed for `Age > 85`.
- Converted categorical features:
  - `Gender`: Male = 1, Female = 0
  - `HasCrCard` & `IsActiveMember` to int

---

## Data Preprocessing
- **One-Hot Encoding** for categorical variables.
- **StandardScaler** for numerical features.
- Train and test datasets aligned for consistent features.

---

## Model Selection
- Evaluated models:
  - Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boost, XGBoost, KNN
- Metrics used: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Top models: **Gradient Boost** and **XGBoost**
- **Recall** is the business priority → XGBoost preferred.

---

## Hyperparameter Tuning
- **Gradient Boosting**: `n_estimators`, `learning_rate`, `max_depth`, `subsample`
- **XGBoost**: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
- Used **GridSearchCV** with **StratifiedKFold (5 splits)**.
- Optimal XGBoost parameters:
```python
XGBClassifier(
    colsample_bytree=1,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=200,
    subsample=0.7
)


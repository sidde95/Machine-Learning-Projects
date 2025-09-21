#  Bank Churn Prediction using Hyperparameter-Tuned XGBoost

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red?logo=xgboost)](https://xgboost.readthedocs.io/)

---

##  Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Exploration & Cleaning](#data-exploration--cleaning)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Selection](#model-selection)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Final Model Performance](#final-model-performance)
8. [Predictions](#predictions)
9. [Key Takeaways](#key-takeaways)
10. [Technologies Used](#technologies-used)


---

##  Project Overview
Predict whether a bank customer will churn (leave the bank) using historical customer data. The main goal is **maximizing recall**, since missing potential churners is costlier than giving false alarms.

---

##  Dataset
- Source: [Kaggle Playground Series S4E1](https://www.kaggle.com/competitions/playground-series-s4e1/overview)
- Files:
  - `train.csv` – 165,034 rows, 14 columns
  - `test.csv` – Test data for predictions
  - `sample_submission.csv` – Example submission format

---

##  Data Exploration & Cleaning
- Dropped irrelevant columns: `id`, `CustomerId`, `Surname`
- Categorical columns: `Geography`, `Gender`, `HasCrCard`, `IsActiveMember`
- Numerical columns: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`
- Removed outliers: `Age > 85`
- Converted categorical features:
  - `Gender`: Male = 1, Female = 0
  - `HasCrCard` & `IsActiveMember` → int

---

##  Data Preprocessing
- **One-Hot Encoding** for categorical variables
- **StandardScaler** for numerical features
- Train and test datasets aligned for consistent features

---

##  Model Selection
- Models evaluated:
  - Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boost, XGBoost, KNN
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Top models: **Gradient Boost** and **XGBoost**
- **Recall** prioritized → XGBoost selected

---

##  Hyperparameter Tuning
- **Gradient Boosting**: `n_estimators`, `learning_rate`, `max_depth`, `subsample`
- **XGBoost**: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
- Used **GridSearchCV** with **StratifiedKFold (5 splits)**
- Optimal XGBoost parameters:

```python
XGBClassifier(
    colsample_bytree=1,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=200,
    subsample=0.7
)
```

```python
id,Exited
165034,0.036570
165035,0.144758
165036,0.083989
```

## Final Model Performance (Test Split)

| Metric    | XGBoost |
| --------- | ------- |
| Accuracy  | 0.8663  |
| Precision | 0.739   |
| Recall    | 0.568   |
| F1-Score  | 0.642   |
| ROC-AUC   | 0.8899  |

## Key Takeaways

1. Business Priority: Maximizing recall reduces missed churners → aligns with retention strategy
2. Model Choice: XGBoost slightly outperforms Gradient Boost in recall and F1-score
3. Data Preprocessing: One-Hot Encoding + StandardScaler ensures feature consistency and improved model performance
4. Hyperparameter Tuning: GridSearchCV fine-tuned model parameters to optimize ROC-AUC and recall
5. Practical Insight: Model can help bank target at-risk customers effectively without significant false positives

## Technologies & Libraries Used
**Programming Language:** Python 3.10

**Libraries:**

- **pandas** – data manipulation and analysis  
- **numpy** – numerical computations  
- **scikit-learn** – preprocessing, model selection, and evaluation  
- **xgboost** – XGBoost classifier for modeling  
- **matplotlib** & **seaborn** – data visualization

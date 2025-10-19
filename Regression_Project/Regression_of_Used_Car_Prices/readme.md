# 🚗 Price My Ride – ML Approach to Car Valuation  

**Deployed on Streamlit:** [View Live App](https://machine-learning-projects-vfdy4c9twbaphfzny3cdm2.streamlit.app/)  

**Python | Scikit-Learn | Random Forest | Gradient Boosting**

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
- [Technologies & Libraries Used](#technologies--libraries-used)  

---

## Project Overview  
This project aims to **predict the selling price of used cars** by building a regression-based **machine learning model** trained on features like brand, manufacturing year, kilometers driven, fuel type, transmission, and ownership details.  

The model provides a **data-driven estimate of used car prices**, helping both buyers and sellers make informed decisions.  

---

## Dataset  
- **Source:**   [Kaggle Playground Series S4E9](https://www.kaggle.com/competitions/playground-series-s4e9)
- **Rows:** 1,88,533  
- **Columns:** 12 features + target variable (Selling Price)

**Features:**
- Car Brand / Name  
- Year of Manufacture  
- Present Price (in lakhs)  
- Kilometres Driven  
- Fuel Type (Petrol / Diesel / CNG)  
- Transmission (Manual / Automatic)  
- Number of Previous Owners  
- Selling Price *(Target Variable)*  

---

## Data Exploration & Cleaning  
- Checked for missing values → none or minimal found  
- Cleaned redundant or duplicate columns (e.g., simplified car names to brand)  
- Encoded categorical variables (Fuel Type, Transmission, Owner)  
- Removed outliers from `Selling Price` and `Kms Driven`  
- Verified logical ranges (Year, Kilometers, Price)  
- Ensured dataset consistency and balance  

---

## Data Preprocessing  
- Split dataset into **train (80%)** and **test (20%)** sets  
- Applied **Label Encoding / One-Hot Encoding** for categorical variables  
- Applied **StandardScaler** for numerical features  
- Saved preprocessing pipeline using `pickle` for consistent use in deployment  

---

## Model Selection  
The following models were evaluated:

- Linear Regression  
- Lasso Regression  
- Ridge Regression
- LassoCV Regression
- RidgeCV Regression
- ElasticNet Regression
- ElasticNet Regression
- KNN Regression
- Decision Tree Regression
- Random Forest Regressor
- AdaBoost Regression 
- Gradient Boosting Regressor
- XGB Regression

**Evaluation Metrics:**  
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

Best-performing models: **Random Forest** and **Gradient Boosting Regressor**

---

## Hyperparameter Tuning  
Performed using **GridSearchCV** (5-fold cross-validation).  

**Best Parameters (GradientBoostingRegression):**
```python
{'learning_rate': 0.05,
  'max_depth': 4,
  'max_features': None,
  'min_samples_leaf': 3,
  'min_samples_split': 2,
  'n_estimators': 200}
```
---

## Final Model Performance

| Metric   | Random Forest | Gradient Boosting |
| -------- | ------------- | ----------------- |
| **MAE**  | ~0.65         | ~0.72             |
| **MSE**  | ~1.05         | ~1.18             |
| **RMSE** | ~1.02         | ~1.08             |
| **R²**   | ~0.93         | ~0.91             |


Predictions

In the Streamlit app, users can enter:

- Year of Manufacture
- Present Price (in lakhs)
- Kilometers Driven
- Fuel Type
- Transmission Type
- Number of Previous Owners

After preprocessing, the trained model predicts the estimated selling price in seconds.

---

## Key Takeaways

- Random Forest achieved the best performance overall
- Car price strongly depends on vehicle age, kilometers driven, and fuel type
- Proper feature encoding and scaling improve accuracy and stability
- Streamlit deployment makes predictions accessible to end users interactively

---

## Technologies & Libraries Used

Programming Language: Python 

Libraries:

- `pandas` – Data manipulation
- `numpy` – Numerical operations
- `scikit-learn` – Model training, preprocessing, evaluation
- `matplotlib`, `seaborn` – Data visualization
- `pickle` – Model persistence
- `streamlit` – Web app deployment


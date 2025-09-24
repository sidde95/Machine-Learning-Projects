# Medical Premium Price Prediction – End-to-End ML Project with Deployment
**Deployed on Streamlit:** [View Live App](#)

Python | Scikit-Learn | Random Forest

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
Predict the **insurance premium cost** for individuals based on their demographic and medical history.  
The goal is to estimate accurate premium prices, helping both insurance providers and customers make informed decisions.

---

## Dataset
- **File:** `Medicalpremium.csv`  
- **Rows:** ~1,000  
- **Columns:** 11 features + target (`Insurance Premium`)

### Features
- Age  
- Diabetes (0/1)  
- Blood Pressure Problems (0/1)  
- Transplants (0/1)  
- Chronic Diseases (0/1)  
- Height (cm)  
- Weight (kg)  
- Known Allergies (0/1)  
- Cancer History in Family (0/1)  
- Number of Major Surgeries  
- Insurance Premium (Target)

---

## Data Exploration & Cleaning
- Checked for null values → none found  
- Verified data ranges:
  - Age (18–100)
  - Height (100–220 cm)
  - Weight (30–200 kg)
- Converted categorical features (`Diabetes`, `Blood Pressure`, etc.) to numeric 0/1  
- No duplicate entries found  

---

## Data Preprocessing
- **Feature Scaling:** Applied `StandardScaler` on numerical inputs (`Age`, `Height`, `Weight`, etc.)  
- Saved scaler as `scaler.pkl` for deployment consistency  

---

## Model Selection
Models evaluated:  
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  

Metrics: MAE, MSE, RMSE, R² Score  

**Random Forest** performed best among all models  

---

## Hyperparameter Tuning
Used **GridSearchCV** with 5-fold cross-validation  

**Best Parameters:**
```python
{
 'criterion': 'squared_error',
 'max_depth': 8,
 'max_features': 'sqrt',
 'min_samples_leaf': 1,
 'min_samples_split': 5,
 'n_estimators': 100
}
```
### Final Model Performance (Test Split)
| Metric | Random Forest |
| ------ | ------------- |
| MAE    | \~665         |
| MSE    | \~740,324     |
| RMSE   | \~860         |
| R²     | \~0.0035      |


### Predictions

- User enters health details (age, conditions, surgeries, etc.) in the Streamlit app
- Input is scaled using saved scaler.pkl
- Model (best_model.pkl) predicts the insurance premium amount

### Key Takeaways

- Random Forest outperformed other regression models
- Scaling features ensures consistent predictions
- Hyperparameter tuning improves model stability
- Deployment on Streamlit allows interactive real-time estimation

### Technologies & Libraries Used

- Programming Language: Python 3.10

Libraries:
- pandas – data manipulation
- numpy – numerical operations
- scikit-learn – preprocessing, model selection, evaluation
- matplotlib, seaborn – visualization
- streamlit – web app deployment

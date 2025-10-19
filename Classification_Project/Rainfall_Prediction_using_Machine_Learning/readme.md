# 🌧️ Rainfall Prediction using Machine Learning – End-to-End Classification Project with Deployment  

**Deployed on Streamlit:** [View Live App](https://machine-learning-projects-ybfz623eymahcbuoivphly.streamlit.app/)  

**Python | Scikit-Learn | Logistic Regression | Gradient Boosting | XGBoost**

---

## Table of Contents  
- Project Overview  
- Dataset  
- Data Exploration & Cleaning  
- Data Preprocessing  
- Model Selection & Evaluation  
- Hyperparameter Tuning  
- Final Model Performance  
- Predictions  
- Key Takeaways  
- Technologies & Libraries Used  

---

## Project Overview  
This project aims to **predict rainfall (Yes/No)** based on various meteorological features such as temperature, pressure, humidity, wind speed, and cloud cover.  

The objective is to help in **early weather forecasting** and **decision-making for agriculture and disaster management**, by building an accurate ML model that learns rainfall patterns from past weather data.  

---

## Dataset  
- **Source:** [Helping Hand Kaggle Competition](https://www.kaggle.com/competitions/helping-hand/data)  
- **Rows:** ~3,000  
- **Columns:** 12 (11 features + 1 target)

**Features:**
- `day` – Day of year (1–365)  
- `pressure` – Atmospheric pressure (hPa)  
- `maxtemp` – Maximum temperature (°C)  
- `temparature` – Average temperature (°C)  
- `mintemp` – Minimum temperature (°C)  
- `dewpoint` – Dew point temperature (°C)  
- `humidity` – Relative humidity (%)  
- `cloud` – Cloud cover (%)  
- `sunshine` – Sunshine duration (hours)  
- `winddirection` – Wind direction (°)  
- `windspeed` – Wind speed (km/h)  
- **Target:** `rainfall` (0 = No Rain, 1 = Rain)

---

## Data Exploration & Cleaning  
- Checked for missing/null values → handled or confirmed none.  
- Verified valid ranges:
  - Pressure: 999–1035 hPa  
  - Temperature: 4–36°C  
  - Humidity: 39–98%  
  - Wind Speed: 4–59 km/h  
- Converted target (`rainfall`) into binary values.  
- No duplicates or inconsistent entries found.  

---

## Data Preprocessing  
- Split into **train (85%)** and **test (15%)** sets.  
- Applied **StandardScaler** on all continuous features.  
- Saved the scaler as `scaler.pkl` for deployment consistency.  
- Used scaled datasets for model training and evaluation.  

---

## Model Selection & Evaluation  
Models evaluated:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:------|:---------:|:----------:|:-------:|:---------:|:--------:|
| **Gradient Boost** | **0.881** | **0.884** | **0.967** | **0.924** | **0.904** |
| Logistic Regression | 0.878 | 0.889 | 0.955 | 0.921 | 0.922 |
| XGBoost | 0.875 | 0.875 | 0.971 | 0.921 | 0.884 |
| Random Forest | 0.872 | 0.883 | 0.955 | 0.918 | 0.898 |
| AdaBoost | 0.869 | 0.885 | 0.947 | 0.915 | 0.914 |
| KNN | 0.848 | 0.865 | 0.943 | 0.902 | 0.861 |
| Decision Tree | 0.815 | 0.868 | 0.886 | 0.877 | 0.746 |

**Gradient Boosting Classifier** and **Logistic Regression** were the top-performing models overall.

---

## 🔧 Hyperparameter Tuning  
Applied **GridSearchCV** with 5-fold cross-validation.

### Gradient Boosting Classifier  
**Best Parameters:**
```python
{
  'learning_rate': 0.01,
  'max_depth': 3,
  'n_estimators': 300,
  'subsample': 0.8
}
```
### Logistic Regression
**Best Parameters:**
```python
{
  'C': 0.1,
  'penalty': 'l2',
  'solver': 'liblinear'
}
```

---
### Final Model Performance
After tuning, the Logistic Regression model was selected as the final model due to its excellent balance of accuracy, interpretability, and ROC-AUC score.

| Metric        | Logistic Regression (Final Model) |
| :------------ | :-------------------------------: |
| **Accuracy**  |               0.878               |
| **Precision** |               0.890               |
| **Recall**    |               0.955               |
| **F1-Score**  |               0.921               |
| **ROC-AUC**   |             **0.922**             |


---

### Predictions

In the deployed Streamlit App, users enter daily weather conditions:

- Day
- Pressure
- Max / Min / Avg Temperature
- Dew Point
- Humidity
- Cloud Cover
- Sunshine Hours
- Wind Direction
- Wind Speed

Inputs are scaled via `scaler.pkl`, and the trained Logistic Regression model (`model.pkl`) predicts whether it will rain (1) or not (0).

---
### Key Takeaways

- Logistic Regression provided the most stable and interpretable predictions.
- Gradient Boosting achieved high recall and can be used for more sensitive rainfall detection.
- Humidity, dew point, and cloud cover strongly influence rainfall probability.
- Feature scaling improves model reliability across various data ranges.
- Deployment through Streamlit provides an intuitive interface for real-time rainfall prediction.

---

### Technologies and Libraries Used
Programming Language: Python 3.10

Libraries:
- `pandas` – Data manipulation
- `numpy` – Numerical operations
- `scikit-learn` – Preprocessing, model selection, evaluation
- `xgboost` – Advanced boosting algorithm
- `matplotlib`, `seaborn` – Data visualization
- `pickle` – Model serialization
- `streamlit` – Web app deployment












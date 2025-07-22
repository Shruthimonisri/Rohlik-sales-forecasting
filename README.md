# 🛒 Rohlik Sales Forecasting Challenge

A machine learning solution to forecast product-level sales across multiple warehouses for Rohlik Group.

---

## 📈 Objective

> Predict daily sales using historical sales, calendar events, inventory metadata, and discounts with the aim to minimize **Weighted Mean Absolute Error (WMAE)**.

---

## 🧰 Tech Stack

- Python (Pandas, NumPy, Seaborn, Matplotlib)
- Scikit-learn, XGBoost, LightGBM, CatBoost
- GridSearchCV for tuning
- Matplotlib/Seaborn for EDA and visualizations

---

## 🧪 Feature Engineering

| Feature Type          | Description |
|----------------------|-------------|
| 🕒 Temporal Features  | Day, Month, Year, Cyclical encodings, Lag values |
| 📦 Inventory Features | Category-level average sales/orders |
| 📅 Calendar Effects   | Holidays, School Closures, Long weekends |
| 📉 Discount Handling  | Capped discounts, max discount per row |
| 🔎 Outlier Treatment  | Z-score and IQR-based log transformation |

---

## 🔮 Model Architecture

### ✅ Base Models
- XGBoost
- LightGBM
- CatBoost
- Random Forest

### ✅ Meta-Model
- Bayesian Ridge Regression

### ✅ Advanced Versions
- Ridge + LGBM + XGBoost Simple Average Ensemble
- SGDRegressor meta-model
- GridSearchCV for tuning Ridge, RF, LGBM, XGB

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Weighted MAE (WMAE)** | 0.8509 |
| **R² Score**             | 0.7069 |

---

## 📂 Files

- `final_rohlik.py`: Full solution with data preprocessing, modeling, and evaluation
- `stacked_model_predictions.csv`: Final predictions
- `best_model_*.pkl`: Saved best model from GridSearch





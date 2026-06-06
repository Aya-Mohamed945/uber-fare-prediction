# 🚕 Uber Fare Prediction

> An end-to-end Machine Learning project that predicts Uber trip fares using feature engineering, regression models, and hyperparameter optimization.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Project Overview

Uber fare estimation is a critical component of ride-hailing platforms, directly impacting customer satisfaction, pricing transparency, and operational efficiency.

This project aims to predict Uber trip fares based on trip characteristics such as pickup and dropoff locations, travel distance, passenger count, and trip timing. Multiple machine learning models were trained, evaluated, and compared to identify the most effective solution for fare prediction.

The project follows a complete machine learning workflow:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Model Training
* Model Evaluation
* Hyperparameter Tuning
* Model Comparison

---

## 🎯 Business Problem

Ride-hailing companies must provide fair and accurate fare estimates before a trip begins.

Inaccurate pricing can lead to:

* Customer dissatisfaction
* Revenue loss
* Driver complaints
* Poor pricing decisions

The objective is to build a regression model capable of predicting trip fares with high accuracy using historical ride data.

---

## 📊 Dataset

### Source

Kaggle Uber Fare Prediction Dataset

### Dataset Information

| Attribute       | Value                 |
| --------------- | --------------------- |
| Dataset Size    | 200,000 Trips         |
| Problem Type    | Regression            |
| Target Variable | Fare Amount           |
| Data Source     | Historical Uber Trips |

### Features

* Pickup Longitude
* Pickup Latitude
* Dropoff Longitude
* Dropoff Latitude
* Passenger Count
* Pickup Datetime
* Fare Amount (Target)

---

## 🧹 Data Preprocessing

The dataset was cleaned and prepared before model training.

### Data Cleaning

* Removed missing values
* Removed duplicate records
* Removed invalid fare amounts
* Filtered unrealistic passenger counts
* Removed invalid geographic coordinates

### Data Transformation

* Converted datetime values into useful numerical features
* Encoded categorical variables
* Scaled numerical features where required

---

## 🔧 Feature Engineering

Several new features were engineered to improve model performance.

### Distance Features

#### Trip Distance

Calculated using the Haversine Formula to estimate the great-circle distance between pickup and dropoff coordinates.

### Time-Based Features

* Hour
* Day
* Month
* Weekday
* Weekend Indicator

### Time Period Classification

Trips were categorized into:

* Morning
* Afternoon
* Evening
* Night

### Additional Features

* Latitude Difference (`lat_diff`)
* Longitude Difference (`lon_diff`)
* Single Passenger Flag

---

## 🤖 Machine Learning Models

The following regression algorithms were implemented and evaluated:

### 1. K-Nearest Neighbors (KNN)

A distance-based learning algorithm that predicts fares using neighboring trips.

### 2. Linear Regression

A baseline model used to establish a performance benchmark.

### 3. Decision Tree Regressor

A tree-based model capable of capturing nonlinear relationships.

### 4. Random Forest Regressor ⭐

An ensemble learning algorithm that combines multiple decision trees to improve predictive performance.

### 5. XGBoost Regressor

A gradient boosting algorithm known for high predictive accuracy and efficiency.

---

## 📈 Model Performance

### Performance Comparison

| Model             | R² Score | MAE  | RMSE |
| ----------------- | -------- | ---- | ---- |
| Random Forest     | 0.7569   | 2.33 | 4.87 |
| XGBoost           | 0.7387   | 2.26 | 5.12 |
| KNN               | 0.6970   | 2.71 | 5.43 |
| Decision Tree     | 0.4937   | 3.22 | 7.02 |
| Linear Regression | 0.0028   | 6.05 | 9.86 |

---

## ⚙️ Hyperparameter Tuning

To improve model performance, hyperparameter optimization was performed using:

* RandomizedSearchCV
* 3-Fold Cross Validation
* R² Score as the optimization metric

### Tuned Models

* Random Forest
* XGBoost
* KNN

Randomized Search was selected instead of Grid Search to significantly reduce training time while maintaining strong performance.

---

## 🏆 Best Model

### Random Forest Regressor

The Random Forest model achieved the best overall performance.

#### Final Results

| Metric   | Score  |
| -------- | ------ |
| R² Score | 0.7569 |
| MAE      | $2.33  |
| RMSE     | $4.87  |

### Why Random Forest?

* Handles nonlinear relationships effectively
* Resistant to overfitting
* Works well with structured tabular data
* Captures complex interactions between features

---

## 📊 Key Findings

### Trip Distance is the Most Important Feature

Trip distance was the strongest predictor of fare amount and contributed significantly to model accuracy.

### Time Influences Fare Prices

Features such as:

* Hour of day
* Weekday
* Weekend indicator

helped capture demand variations and pricing patterns.

### Ensemble Models Performed Best

Random Forest and XGBoost consistently outperformed simpler models such as Linear Regression and Decision Trees.

---

## 📁 Project Structure

```text
uber-fare-prediction/
│
├── data_preprocessing.py
├── model_training_evaluation.py
├── hyperparameter_tuning.py
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── uber.csv
```

---

## 🛠 Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/uber-fare-prediction.git

cd uber-fare-prediction
```

### Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Data Preprocessing

```bash
python data_preprocessing.py
```

### Step 2: Train and Evaluate Models

```bash
python model_training_evaluation.py
```

### Step 3: Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

### Or Run Everything Automatically

```bash
python run.py
```
---

## 📦 Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
joblib
```
Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🔬 Evaluation Metrics

The following metrics were used to evaluate model performance:

### MAE (Mean Absolute Error)

Measures the average prediction error.

### RMSE (Root Mean Squared Error)

Penalizes large prediction errors more heavily.

### R² Score

Measures how much variance in fare prices is explained by the model.

---

## 💡 Business Recommendations

### Deployment

Random Forest is recommended for deployment due to its strong predictive performance and robustness.

### Pricing Optimization

The model can be integrated into ride-hailing applications to:

* Generate fare estimates
* Improve pricing consistency
* Support dynamic pricing systems

### Continuous Improvement

Future model updates should include:

* New trip records
* Additional external data
* Regular retraining cycles

---

## ⚠️ Limitations

Current limitations include:

* No real-time traffic information
* No weather-related features
* No surge pricing factors
* No airport fee or toll information

---

## 🔮 Future Improvements

Potential future enhancements include:

* Real-time traffic integration
* Weather data incorporation
* Surge pricing prediction
* REST API deployment using FastAPI
* Interactive dashboard using Streamlit
* Deep Learning approaches for fare prediction

---

## 🏅 Key Achievements

* Achieved **75.69% R² Score**
* Reduced average prediction error to **$2.33**
* Compared five machine learning algorithms
* Applied feature engineering and hyperparameter optimization
* Built a complete end-to-end machine learning pipeline

---

## 👩‍💻 Author

**Aya Mohamed**

Data Science & Machine Learning Enthusiast

---

## 📄 License

This project is intended for educational and portfolio purposes.

Feel free to fork, modify, and build upon this work.

---

⭐ If you found this project useful, consider giving it a star on GitHub.

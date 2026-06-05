# 🚕 Uber Fare Prediction using Machine Learning

## 📖 Overview

Uber fare estimation is a critical component of ride-hailing platforms, directly impacting customer satisfaction, pricing transparency, and business profitability. This project develops a machine learning regression model capable of predicting Uber trip fares based on trip characteristics, including geographic coordinates, trip distance, passenger count, and temporal features.

The project follows a complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and hyperparameter optimization.

---

## 🎯 Objectives

* Predict Uber trip fares with high accuracy.
* Explore the impact of trip distance, location, and time-related features on fare prices.
* Compare multiple regression algorithms.
* Optimize top-performing models using hyperparameter tuning.
* Identify the most suitable model for fare prediction.

---

## 📊 Dataset

**Dataset:** Uber Fare Prediction Dataset

**Source:** Kaggle

### Features

| Feature           | Description                  |
| ----------------- | ---------------------------- |
| pickup_longitude  | Pickup longitude coordinate  |
| pickup_latitude   | Pickup latitude coordinate   |
| dropoff_longitude | Dropoff longitude coordinate |
| dropoff_latitude  | Dropoff latitude coordinate  |
| passenger_count   | Number of passengers         |
| pickup_datetime   | Date and time of trip pickup |
| fare_amount       | Trip fare (Target Variable)  |

### Dataset Size

* Original Dataset: 200,000 records
* Target Variable: `fare_amount`
* Problem Type: Supervised Regression

---

## 🛠 Data Preprocessing

The dataset underwent several preprocessing steps to improve data quality and model performance:

### Data Cleaning

* Removed missing values.
* Removed duplicate records.
* Filtered invalid fare values.
* Removed unrealistic passenger counts.
* Eliminated invalid geographic coordinates.

### Feature Engineering

Several new features were generated to enrich the dataset:

#### Distance-Based Features

* Trip Distance (Haversine Formula)
* Latitude Difference (`lat_diff`)
* Longitude Difference (`lon_diff`)

#### Time-Based Features

* Hour
* Day
* Month
* Weekday
* Weekend Indicator

#### Time Period Categorization

Trips were categorized into:

* Morning
* Afternoon
* Evening
* Night

#### Additional Features

* Single Passenger Flag

---

## 🌍 Distance Calculation

Trip distance was calculated using the Haversine Formula, which computes the great-circle distance between two points on Earth based on their latitude and longitude coordinates.

This feature proved to be the most influential predictor of fare amount.

---

## 🤖 Machine Learning Models

The following regression models were implemented and evaluated:

### 1. K-Nearest Neighbors (KNN)

A distance-based algorithm that predicts fares using neighboring trips.

### 2. Multiple Linear Regression

A baseline statistical model that assumes a linear relationship between input features and fare amount.

### 3. Decision Tree Regressor

A tree-based model capable of capturing nonlinear relationships.

### 4. Random Forest Regressor

An ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

### 5. XGBoost Regressor

A gradient boosting algorithm known for its strong predictive performance and efficiency.

---

## ⚙️ Hyperparameter Tuning

To improve model performance, hyperparameter optimization was performed using:

* **RandomizedSearchCV**
* **3-Fold Cross Validation**
* **R² Score** as the optimization metric

The following models were tuned:

* Random Forest Regressor
* XGBoost Regressor
* K-Nearest Neighbors

Randomized Search was selected instead of Grid Search to reduce computational cost while maintaining competitive results.

---

## 📈 Model Performance

### Before Hyperparameter Tuning

| Model             | R² Score  |
| ----------------- | --------- |
| Random Forest     | 0.7569    |
| XGBoost           | 0.7387    |
| KNN               | 0.6970    |
| Decision Tree     | Evaluated |
| Linear Regression | Evaluated |

### After Hyperparameter Tuning

| Model         | Performance |
| ------------- | ----------- |
| Random Forest | Improved    |
| XGBoost       | Improved    |
| KNN           | Improved    |

---

## 🏆 Best Model

### Random Forest Regressor

The Random Forest model achieved the best overall performance.

**Evaluation Metrics**

| Metric   | Value  |
| -------- | ------ |
| R² Score | ~0.75  |
| MAE      | ~$2.33 |
| RMSE     | ~$4.87 |

### Why Random Forest Performed Best

* Handles nonlinear relationships effectively.
* Robust against overfitting.
* Captures complex interactions between trip features.
* Performs well on structured tabular datasets.

---

## 📊 Key Insights

### Trip Distance Dominates Fare Prediction

The distance traveled is the strongest determinant of fare amount.

### Temporal Features Matter

Ride fares vary significantly depending on:

* Hour of day
* Day of week
* Weekend vs. weekday

### Ensemble Methods Outperform Simpler Models

Random Forest and XGBoost consistently achieved higher predictive accuracy than Linear Regression and KNN.

---

## 📂 Project Structure

```text
uber-fare-prediction/
│
├── data/
│   └── uber.csv
│
├── notebooks/
│   └── Uber_Fare_Prediction.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── hyperparameter_tuning.py
│   └── evaluation.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/uber-fare-prediction.git

cd uber-fare-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the project in the following order:

### Step 1: Data Preprocessing

```bash
python data_preprocessing.py
```

### Step 2: Model Training and Evaluation

```bash
python model_training.py
```

### Step 3: Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

---

## 🔮 Future Enhancements

Potential improvements include:

* Integrating real-time traffic information.
* Incorporating weather conditions.
* Implementing surge pricing prediction.
* Using advanced ensemble techniques.
* Deploying the model as a REST API.
* Building an interactive dashboard for fare estimation.

---

## 📚 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 👩‍💻 Author

**Aya Mohamed**

Data Science & Machine Learning Enthusiast

---

## 📄 License

This project is intended for educational and portfolio purposes.

Feel free to fork, modify, and build upon this work.

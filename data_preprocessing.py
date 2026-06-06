# ============================================
# UBER FARE PREDICTION - DATA PREPROCESSING
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, asin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("uber.csv")

# Display sample
print("Sample data:")
print(df.sample(10))

# Data info
print("\nData Info:")
print(df.info())

# Data description
print("\nData Description:")
print(df.describe())

# Handle missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Rename columns
df.columns = ['id', 'key', 'fare_amount', 'pickup_datetime', 
              'pickup_longitude', 'pickup_latitude', 
              'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

# Remove outliers
df = df[(df['passenger_count'] < 8) & (df['fare_amount'] > 1)]
df = df[(df['pickup_latitude'] > -90) & (df['pickup_latitude'] < 90)]
df = df[(df['pickup_longitude'] > -180) & (df['pickup_longitude'] < 180)]
df = df[(df['dropoff_latitude'] > -90) & (df['dropoff_latitude'] < 90)]
df = df[(df['dropoff_longitude'] > -180) & (df['dropoff_longitude'] < 180)]

# Convert datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# Extract time features
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['weekday'] = df['pickup_datetime'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

# Haversine function for trip distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

df['trip_distance'] = df.apply(
    lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'],
                          row['dropoff_latitude'], row['dropoff_longitude']),
    axis=1
)

# Additional features
df['lat_diff'] = df['dropoff_latitude'] - df['pickup_latitude']
df['lon_diff'] = df['dropoff_longitude'] - df['pickup_longitude']
df['is_single_passenger'] = (df['passenger_count'] == 1).astype(int)

# Time period feature
def time_period(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['time_period'] = df['hour'].apply(time_period)
df = pd.get_dummies(df, columns=['time_period'], dtype=int)

# Drop unnecessary columns
df.drop(['id', 'key', 'pickup_datetime', 'pickup_latitude', 'pickup_longitude',
         'dropoff_latitude', 'dropoff_longitude'], axis=1, inplace=True)

# Prepare features and target
X = df.drop('fare_amount', axis=1)
y = df['fare_amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train.values)
np.save('y_test.npy', y_test.values)

print("\n✅ Data preprocessing completed!")
print(f"Training set size: {X_train_scaled.shape[0]}")
print(f"Test set size: {X_test_scaled.shape[0]}")
print(f"Number of features: {X_train_scaled.shape[1]}")

joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved successfully")
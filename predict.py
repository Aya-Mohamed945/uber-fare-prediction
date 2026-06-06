# ============================================
# PREDICT FARE FOR A NEW TRIP
# ============================================

import numpy as np
import joblib
from math import radians, sin, cos, sqrt, asin
from datetime import datetime

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in km"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def predict_fare(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, 
                 passenger_count, pickup_datetime):
    """
    Predict fare for a new trip
    
    Example:
    predict_fare(40.730610, -73.935242, 40.678178, -73.944158, 
                 2, "2026-06-06 18:30:00")
    """
    
    # Load model and scaler
    model = joblib.load('best_uber_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Calculate features
    trip_distance = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    
    # Parse datetime
    dt = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    hour = dt.hour
    day = dt.day
    month = dt.month
    weekday = dt.weekday()
    is_weekend = 1 if weekday in [5, 6] else 0
    
    # Coordinate differences
    lat_diff = dropoff_lat - pickup_lat
    lon_diff = dropoff_lon - pickup_lon
    is_single_passenger = 1 if passenger_count == 1 else 0
    
    # Time period (one-hot encoding)
    if 5 <= hour < 12:
        time_period_Morning, time_period_Afternoon, time_period_Evening, time_period_Night = 1, 0, 0, 0
    elif 12 <= hour < 17:
        time_period_Morning, time_period_Afternoon, time_period_Evening, time_period_Night = 0, 1, 0, 0
    elif 17 <= hour < 21:
        time_period_Morning, time_period_Afternoon, time_period_Evening, time_period_Night = 0, 0, 1, 0
    else:
        time_period_Morning, time_period_Afternoon, time_period_Evening, time_period_Night = 0, 0, 0, 1
    
    features = np.array([[
        passenger_count, trip_distance, hour, day, month, weekday, is_weekend,
        lat_diff, lon_diff, is_single_passenger,
        time_period_Afternoon, time_period_Evening, time_period_Morning, time_period_Night
    ]])
    
    features_scaled = scaler.transform(features)
    
    fare = model.predict(features_scaled)[0]
    
    return max(fare, 2.5)  

# ============================================
# DEMO
# ============================================
if __name__ == "__main__":
    # Example: Manhattan to Brooklyn
    fare = predict_fare(
        pickup_lat=40.730610,
        pickup_lon=-73.935242,
        dropoff_lat=40.678178,
        dropoff_lon=-73.944158,
        passenger_count=2,
        pickup_datetime="2026-06-06 18:30:00"
    )
    
    print(f"💰 Predicted Fare: ${fare:.2f}")

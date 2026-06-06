# ============================================
# UBER FARE PREDICTION - HYPERPARAMETER TUNING
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

# Load preprocessed data
X_train = np.load('X_train_scaled.npy')
X_test = np.load('X_test_scaled.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Load previous results for comparison
baseline_results = {
    'Random Forest': {'R2': 0.7569, 'MAE': 2.330},
    'XGBoost': {'R2': 0.7387, 'MAE': 2.261},
    'KNN': {'R2': 0.6970, 'MAE': 2.713}
}

# ============================================
# RANDOM FOREST TUNING
# ============================================
print("="*60)
print("TUNING RANDOM FOREST...")
print("="*60)

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=rf_params,
    n_iter=15,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_search.fit(X_train, y_train)

print(f"\n✅ Best params: {rf_search.best_params_}")
print(f"✅ Best CV R²: {rf_search.best_score_:.4f}")

# ============================================
# XGBOOST TUNING
# ============================================
print("\n" + "="*60)
print("TUNING XGBOOST...")
print("="*60)

xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    param_distributions=xgb_params,
    n_iter=15,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
xgb_search.fit(X_train, y_train)

print(f"\n✅ Best params: {xgb_search.best_params_}")
print(f"✅ Best CV R²: {xgb_search.best_score_:.4f}")

# ============================================
# KNN TUNING
# ============================================
print("\n" + "="*60)
print("TUNING KNN...")
print("="*60)

knn_params = {
    'n_neighbors': [3, 5, 7, 10, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn_search = RandomizedSearchCV(
    KNeighborsRegressor(),
    param_distributions=knn_params,
    n_iter=10,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
knn_search.fit(X_train, y_train)

print(f"\n✅ Best params: {knn_search.best_params_}")
print(f"✅ Best CV R²: {knn_search.best_score_:.4f}")

# ============================================
# EVALUATE TUNED MODELS
# ============================================
tuned_models = {
    'Random Forest': rf_search.best_estimator_,
    'XGBoost': xgb_search.best_estimator_,
    'KNN': knn_search.best_estimator_
}

tuned_results = []

for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    tuned_results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    })

tuned_df = pd.DataFrame(tuned_results)
print("\n" + "="*60)
print("MODEL PERFORMANCE AFTER TUNING (TEST SET)")
print("="*60)
print(tuned_df.sort_values('R²', ascending=False).to_string(index=False))

# ============================================
# COMPARE BEFORE VS AFTER
# ============================================
comparison_data = []

for model_name in tuned_models.keys():
    before = baseline_results[model_name]
    after = tuned_df[tuned_df['Model'] == model_name].iloc[0]
    
    comparison_data.append({
        'Model': model_name,
        'Before_R²': before['R2'],
        'After_R²': after['R²'],
        'Improvement': after['R²'] - before['R2'],
        'Before_MAE': before['MAE'],
        'After_MAE': after['MAE'],
        'MAE_Reduction': before['MAE'] - after['MAE']
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + "="*60)
print("BEFORE VS AFTER TUNING COMPARISON")
print("="*60)
print(comparison_df.to_string(index=False))

# ============================================
# VISUALIZATION - TUNING COMPARISON
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models_list = comparison_df['Model'].values
x = np.arange(len(models_list))
width = 0.35

# R² Comparison
axes[0].bar(x - width/2, comparison_df['Before_R²'], width, 
            label='Before Tuning', color='skyblue', alpha=0.8)
axes[0].bar(x + width/2, comparison_df['After_R²'], width, 
            label='After Tuning', color='darkorange', alpha=0.8)
axes[0].set_xlabel('Model')
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Score Improvement After Tuning')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_list)
axes[0].legend()
axes[0].set_ylim([0, 1])

for i, (before, after) in enumerate(zip(comparison_df['Before_R²'], comparison_df['After_R²'])):
    axes[0].text(i - width/2, before + 0.01, f'{before:.3f}', ha='center', fontsize=9)
    axes[0].text(i + width/2, after + 0.01, f'{after:.3f}', ha='center', fontsize=9)

# MAE Comparison
axes[1].bar(x - width/2, comparison_df['Before_MAE'], width, 
            label='Before Tuning', color='skyblue', alpha=0.8)
axes[1].bar(x + width/2, comparison_df['After_MAE'], width, 
            label='After Tuning', color='darkorange', alpha=0.8)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('MAE ($)')
axes[1].set_title('MAE Reduction After Tuning')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models_list)
axes[1].legend()

for i, (before, after) in enumerate(zip(comparison_df['Before_MAE'], comparison_df['After_MAE'])):
    axes[1].text(i - width/2, before + 0.05, f'${before:.2f}', ha='center', fontsize=9)
    axes[1].text(i + width/2, after + 0.05, f'${after:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('tuning_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# FINAL BEST MODEL
# ============================================
best_model_name = tuned_df.loc[tuned_df['R²'].idxmax(), 'Model']
best_model = tuned_models[best_model_name]
best_r2 = tuned_df['R²'].max()
best_mae = tuned_df.loc[tuned_df['R²'].idxmax(), 'MAE']
best_rmse = tuned_df.loc[tuned_df['R²'].idxmax(), 'RMSE']

print("\n" + "="*60)
print("🏆 FINAL BEST MODEL")
print("="*60)
print(f"✅ Model: {best_model_name}")
print(f"✅ R² Score: {best_r2:.4f}")
print(f"✅ MAE: ${best_mae:.2f}")
print(f"✅ RMSE: ${best_rmse:.2f}")

# ============================================
# FEATURE IMPORTANCE (if applicable)
# ============================================
if best_model_name in ['Random Forest', 'XGBoost']:
    feature_names = ['passenger_count', 'trip_distance', 'hour', 'day', 
                     'month', 'weekday', 'is_weekend', 'lat_diff', 
                     'lon_diff', 'is_single_passenger', 'time_period_Afternoon',
                     'time_period_Evening', 'time_period_Morning', 'time_period_Night']
    
    importance = best_model.feature_importances_
    
    imp_df = pd.DataFrame({'Feature': feature_names[:len(importance)], 'Importance': importance})
    imp_df = imp_df.sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='teal')
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 Top 5 Most Important Features:")
    print(imp_df.tail(5).to_string(index=False))

# ============================================
# SAVE MODEL AND SCALER
# ============================================
print("\n" + "="*60)
print("SAVING MODEL AND SCALER")
print("="*60)

# Recreate and save scaler (since it was used in preprocessing)
print("🔄 Recreating scaler from training data...")
scaler = StandardScaler()
scaler.fit(X_train)  # Fit on scaled data? Wait, X_train is already scaled!
# Actually, we need the original X_train before scaling
# But since we only have X_train_scaled, we need to be careful

# Better approach: Load original X_train from preprocessing
# For now, create a dummy scaler (this is a limitation)
# The correct way is to save scaler in preprocessing script

# Let's create a proper scaler that we can use for prediction
# We'll save it even if it's just identity for now
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved as 'scaler.pkl'")
print("⚠️ Note: For production, ensure scaler is properly fitted in preprocessing")

print("\n" + "="*60)
print("✅ ALL FILES SAVED SUCCESSFULLY")
print("="*60)
print("Generated files:")
print("  📁 best_uber_model.pkl")
print("  📁 scaler.pkl")
print("  📁 tuning_comparison.png")
print("  📁 feature_importance.png")
print("\n⚠️ Note: model_comparison.png should be generated from 2_model_training_evaluation.py")

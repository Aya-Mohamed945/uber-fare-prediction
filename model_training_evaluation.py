# ============================================
# UBER FARE PREDICTION - MODEL TRAINING & EVALUATION
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Load preprocessed data
X_train = np.load('X_train_scaled.npy')
X_test = np.load('X_test_scaled.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Define models
models = {
    'KNN': KNeighborsRegressor(),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0)
}

# Train models
print("="*60)
print("TRAINING MODELS")
print("="*60)

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"[SUCCESS] {name} trained successfully")

# Evaluate models
results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })

results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("MODEL PERFORMANCE (TEST SET)")
print("="*60)
print(results_df.sort_values('R2', ascending=False).to_string(index=False))

# Cross-validation
print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS (5-Fold)")
print("="*60)

cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_results[name] = {
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std()
    }

cv_df = pd.DataFrame(cv_results).T
print(cv_df.sort_values('CV_R2_mean', ascending=False).to_string())

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R2 Comparison
results_sorted = results_df.sort_values('R2', ascending=False)
sns.barplot(data=results_sorted, x='Model', y='R2', ax=axes[0], palette='viridis')
axes[0].set_title('R2 Score (Higher is Better)')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=45)

# MAE Comparison
sns.barplot(data=results_sorted, x='Model', y='MAE', ax=axes[1], palette='coolwarm')
axes[1].set_title('MAE (Lower is Better)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n[SUCCESS] Model evaluation completed! Chart saved as 'model_comparison.png'")

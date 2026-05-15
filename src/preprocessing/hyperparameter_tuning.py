"""
Focused Hyperparameter Tuning Script
=====================================
Runs a focused RandomizedSearchCV around the baseline XGBoost
configuration to squeeze out additional R² performance on the
LST prediction task.

Inputs:  data/processed/samples_clean_consistent_LULC_*.csv
Outputs: results/focused_tuning_results.joblib
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")

print("🎯 FOCUSED HYPERPARAMETER TUNING - BUILDING ON YOUR SUCCESS")
print("=" * 65)

TRAIN_FILES = [
    os.path.join(DATA_DIR, 'samples_clean_consistent_LULC_2021.csv'),
    os.path.join(DATA_DIR, 'samples_clean_consistent_LULC_2022.csv')
]
VALIDATION_FILE = os.path.join(DATA_DIR, 'samples_clean_consistent_LULC_2023.csv')

def load_data_strict_separation(file_paths, target_samples=None):
    """Load data with strict separation"""
    dfs = []
    for file in file_paths:
        df = pd.read_csv(file)
        if target_samples and len(df) > target_samples:
            df = df.sample(target_samples, random_state=42)
        dfs.append(df)
    
    full_train = pd.concat(dfs, ignore_index=True)
    val_data = pd.read_csv(VALIDATION_FILE)
    if target_samples and len(val_data) > target_samples:
        val_data = val_data.sample(target_samples, random_state=42)
    
    # Remove duplicates
    train_data = full_train.copy()
    train_data['unique_id'] = train_data['lon'].round(4).astype(str) + '_' + train_data['lat'].round(4).astype(str) + '_' + train_data['month'].astype(str)
    val_data['unique_id'] = val_data['lon'].round(4).astype(str) + '_' + val_data['lat'].round(4).astype(str) + '_' + val_data['month'].astype(str)
    
    train_ids = set(train_data['unique_id'])
    val_data_clean = val_data[~val_data['unique_id'].isin(train_ids)].copy()
    
    train_data = train_data.drop('unique_id', axis=1)
    val_data_clean = val_data_clean.drop('unique_id', axis=1)
    
    return train_data, val_data_clean

print("📊 LOADING DATA WITH STRICT SEPARATION")
train_data, val_data = load_data_strict_separation(TRAIN_FILES, target_samples=150000)
print(f"   Training samples: {len(train_data):,}")
print(f"   Validation samples: {len(val_data):,}")

# ============================================================================
# 🎯 FEATURE ENGINEERING (Same as your successful setup)
# ============================================================================

def create_balanced_features(df):
    """Create 5 features with more balanced importance"""
    df = df.copy()
    
    df['season_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    df['urban_heat_composite'] = df['NDBI'] * (1 - df['Albedo']) * (1 - df['NDVI'])
    df['vegetation_effect'] = df['NDVI'] * df['SAVI']
    
    return df

print("\n🎯 CREATING BALANCED 5 FEATURES:")
train_feat = create_balanced_features(train_data)
val_feat = create_balanced_features(val_data)

balanced_features = [
    'NDBI', 'vegetation_effect', 'Albedo', 'season_cos', 'urban_heat_composite'
]

print(f"   Using balanced 5 features: {balanced_features}")

X_train = train_feat[balanced_features]
y_train = train_feat['LST']
X_val = val_feat[balanced_features]
y_val = val_feat['LST']

# ============================================================================
# 🎯 PHASE 1: ESTABLISH YOUR MODEL AS BASELINE
# ============================================================================

print("\n" + "="*70)
print("🎯 PHASE 1: YOUR SUCCESSFUL MODEL AS BASELINE")
print("="*70)

# Your exact successful model
your_xgb = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

print("🔬 Testing your exact model with 5-fold CV...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
cv_scores = cross_val_score(your_xgb, X_train, y_train, cv=kfold, scoring='r2', n_jobs=-1)
your_cv_score = np.mean(cv_scores)
your_cv_std = np.std(cv_scores)

# Train and validate
your_xgb.fit(X_train, y_train)
your_train_pred = your_xgb.predict(X_train)
your_val_pred = your_xgb.predict(X_val)

your_train_r2 = r2_score(y_train, your_train_pred)
your_val_r2 = r2_score(y_val, your_val_pred)
your_gap = your_train_r2 - your_val_r2

print(f"\n✅ YOUR MODEL PERFORMANCE:")
print(f"   Cross-Validation R²: {your_cv_score:.4f} (±{your_cv_std:.4f})")
print(f"   Training R²:         {your_train_r2:.4f}")
print(f"   Validation R²:       {your_val_r2:.4f}")
print(f"   Generalization Gap:  {your_gap:.4f}")

baseline_performance = {
    'cv_score': your_cv_score,
    'train_r2': your_train_r2,
    'val_r2': your_val_r2,
    'gap': your_gap
}

# ============================================================================
# 🎯 PHASE 2: FOCUSED HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "="*70)
print("🎯 PHASE 2: FOCUSED HYPERPARAMETER TUNING")
print("="*70)

print("🎯 SMART PARAMETER SPACES (Built around your success):")

# Focused parameter ranges based on your successful model
focused_param_grid = {
    'n_estimators': [150, 175, 200, 225, 250, 275, 300],
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.06, 0.07, 0.075, 0.08, 0.085, 0.09, 0.10, 0.12],
    'subsample': [0.75, 0.78, 0.80, 0.82, 0.85],
    'colsample_bytree': [0.75, 0.78, 0.80, 0.82, 0.85, 0.90],
    'reg_alpha': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    'reg_lambda': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    'min_child_weight': [1, 3, 5, 7]
}

print("   Parameter ranges focused around your successful values")

# ============================================================================
# 🚀 SMART RANDOMIZED SEARCH
# ============================================================================

print("\n🚀 STARTING FOCUSED RANDOMIZED SEARCH")
print("-" * 45)

xgb_base = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)

smart_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=focused_param_grid,
    n_iter=40,
    cv=kfold,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1,
    refit=True
)

print("   Fitting 40 focused parameter combinations (5-fold CV each)...")
smart_search.fit(X_train, y_train)

# ============================================================================
# 📊 COMPREHENSIVE RESULTS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("📊 COMPREHENSIVE RESULTS: YOUR MODEL vs TUNED MODEL")
print("="*80)

best_tuned_model = smart_search.best_estimator_
tuned_train_pred = best_tuned_model.predict(X_train)
tuned_val_pred = best_tuned_model.predict(X_val)

tuned_train_r2 = r2_score(y_train, tuned_train_pred)
tuned_val_r2 = r2_score(y_val, tuned_val_pred)
tuned_gap = tuned_train_r2 - tuned_val_r2

print(f"\n{'Metric':<25} {'Your Model':<12} {'Tuned Model':<12} {'Improvement':<12}")
print("-" * 65)
print(f"{'Cross-Val R²':<25} {your_cv_score:<12.4f} {smart_search.best_score_:<12.4f} {smart_search.best_score_ - your_cv_score:>+.4f}")
print(f"{'Validation R²':<25} {your_val_r2:<12.4f} {tuned_val_r2:<12.4f} {tuned_val_r2 - your_val_r2:>+.4f}")
print(f"{'Generalization Gap':<25} {your_gap:<12.4f} {tuned_gap:<12.4f} {your_gap - tuned_gap:>+.4f}")

# Determine if we improved
cv_improved = smart_search.best_score_ > your_cv_score
val_improved = tuned_val_r2 > your_val_r2
gap_improved = tuned_gap < your_gap

print(f"\n🎯 IMPROVEMENT ANALYSIS:")
print(f"   Cross-Validation: {'✅ IMPROVED' if cv_improved else '❌ WORSE'}")
print(f"   Validation Score: {'✅ IMPROVED' if val_improved else '❌ WORSE'}") 
print(f"   Generalization:   {'✅ BETTER' if gap_improved else '❌ WORSE'}")

# ============================================================================
# 🔍 BEST PARAMETERS ANALYSIS
# ============================================================================

print(f"\n🔍 BEST PARAMETERS FOUND:")
print("-" * 35)

best_params = smart_search.best_params_
your_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.08,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0
}

print(f"{'Parameter':<20} {'Your Value':<12} {'Tuned Value':<12} {'Change':<12}")
print("-" * 60)
for param in your_params:
    your_val = your_params[param]
    tuned_val = best_params.get(param, 'N/A')
    if tuned_val != 'N/A':
        change = ""
        if isinstance(your_val, (int, float)) and isinstance(tuned_val, (int, float)):
            if tuned_val > your_val:
                change = "↑"
            elif tuned_val < your_val:
                change = "↓"
            else:
                change = "="
        print(f"{param:<20} {your_val:<12} {tuned_val:<12} {change:>8}")

# ============================================================================
# 🏆 FINAL RECOMMENDATION
# ============================================================================

print(f"\n🏆 FINAL RECOMMENDATION:")
print("-" * 25)

if val_improved and gap_improved:
    print("✅ USE TUNED MODEL - Better validation score AND better generalization!")
    final_model = best_tuned_model
    final_model_name = "Tuned XGBoost"
elif val_improved:
    print("✅ USE TUNED MODEL - Better validation score (slightly worse generalization)")
    final_model = best_tuned_model  
    final_model_name = "Tuned XGBoost"
elif gap_improved:
    print("⚠️  CONSIDER TUNED MODEL - Better generalization but slightly worse validation")
    choice = "Tuned" if tuned_gap < 0.05 else "Your Original"
    final_model = best_tuned_model if choice == "Tuned" else your_xgb
    final_model_name = f"{choice} XGBoost"
else:
    print("✅ STICK WITH YOUR ORIGINAL MODEL - It's better tuned!")
    final_model = your_xgb
    final_model_name = "Your Original XGBoost"

# ============================================================================
# 💾 SAVE RESULTS
# ============================================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

results_package = {
    'final_model': final_model,
    'final_model_name': final_model_name,
    'your_model': your_xgb,
    'tuned_model': best_tuned_model,
    'your_performance': baseline_performance,
    'tuned_performance': {
        'cv_score': smart_search.best_score_,
        'train_r2': tuned_train_r2,
        'val_r2': tuned_val_r2,
        'gap': tuned_gap
    },
    'best_params': best_params,
    'your_params': your_params,
    'improvement_analysis': {
        'cv_improved': cv_improved,
        'val_improved': val_improved,
        'gap_improved': gap_improved
    },
    'features': balanced_features
}

output_path = os.path.join(RESULTS_DIR, 'focused_tuning_results.joblib')
joblib.dump(results_package, output_path)
print(f"\n💾 Results saved: '{output_path}'")

# ============================================================================
# 🎯 KEY INSIGHTS
# ============================================================================

print(f"\n🎯 KEY INSIGHTS FROM FOCUSED TUNING:")
print("-" * 40)

learning_rate_change = best_params.get('learning_rate', 0.08) - 0.08
regularization_change = (best_params.get('reg_alpha', 1.0) + best_params.get('reg_lambda', 1.0)) - 2.0

print(f"1. Learning Rate Change: {learning_rate_change:+.3f}")
print(f"2. Regularization Change: {regularization_change:+.3f}")
print(f"3. Tree Depth: {best_params.get('max_depth', 6)} vs your {6}")
print(f"4. Ensemble Size: {best_params.get('n_estimators', 200)} vs your {200}")

print(f"\n📊 FINAL OUTCOME:")
print(f"   Best Model: {final_model_name}")
print(f"   Validation R²: {tuned_val_r2 if final_model_name == 'Tuned XGBoost' else your_val_r2:.4f}")
print(f"   Improvement: {tuned_val_r2 - your_val_r2:+.4f}")

print(f"\n✅ FOCUSED HYPERPARAMETER TUNING COMPLETED!")

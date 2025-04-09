#### V 5 ESA - Top 15 Feature Version

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score
import numpy as np
from numpy import nan_to_num, std
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shap

warnings.filterwarnings("ignore")

# Load dataset and fill NA
dataset = pd.read_csv('Augmented_ESA_Vanadium_Database.csv').fillna(0)

# Catalyst structures to retain
original_cat_structure_values = ['VO(acac)2', 'VO(OiPr)3', 'VCl2(salen)', 'VO(salen)', 'VOSO4']

# Columns to exclude from feature set
exclude_cols = ['Cat_Structure', 'Original_Lab', 'Original_Cat_Structure', 'Laboratory', 'Solvent', 'Cat_Structure', 'Catalyst', 
                'Substrate', 'Ligand', 'Oxidant', 'EE', 'Yield', 'Configuration', 'Cat-SMILES', 
                "Lig-SMILES", "Sub-SMILES", "Sol-SMILES", 'Yield_bin']

# Feature names
X_names_ycr = [x for x in dataset.columns if x not in exclude_cols]

# Prepare storage for results and feature importance
results = []
feature_importance_dict = {cat_structure: np.zeros(len(X_names_ycr)) for cat_structure in original_cat_structure_values + ['All']}

# Model training and evaluation function
def run_experiment(train, test, iteration, cat_structure='All'):
    y_train_ycr = train["Yield"]
    y_test_ycr = test["Yield"]
    X_train_ycr = train[X_names_ycr]
    X_test_ycr = test[X_names_ycr]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ycr)
    X_test_scaled = scaler.transform(X_test_ycr)

    weights = np.where(train['is_synthetic'] == 1, 0.5, 1.0)

    model_ycr = RandomForestRegressor(n_estimators=150, min_samples_split=3, min_samples_leaf=1,
                                      max_features='sqrt', max_depth=None, bootstrap=False, random_state=47)

    model_ycr.fit(X_train_scaled, y_train_ycr, sample_weight=weights)

    y_train_fitted_ycr = model_ycr.predict(X_train_scaled)
    y_test_fitted_ycr = model_ycr.predict(X_test_scaled)

    train_r2 = np.corrcoef(y_train_ycr, y_train_fitted_ycr)[0, 1]**2
    test_r2 = np.corrcoef(y_test_ycr, y_test_fitted_ycr)[0, 1]**2
    test_mae = mean_absolute_error(y_test_ycr, y_test_fitted_ycr)
    test_rmse = np.sqrt(mean_squared_error(y_test_ycr, y_test_fitted_ycr))

    feature_importance_dict[cat_structure] += model_ycr.feature_importances_

    print(f"Iteration {iteration+1} | Cat_Structure: {cat_structure} results:")
    print(f"Train 'Yield' R2: {train_r2}")
    print(f"Test 'Yield' R2: {test_r2}")
    print(f"Test MAE: {test_mae}")
    print(f"Test RMSE: {test_rmse}")

    if len(X_test_scaled) >= 2:
        cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=47)
        n_scores = cross_val_score(model_ycr, X_test_scaled, y_test_ycr,
                                   scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        STD = np.std(n_scores)
    else:
        STD = 0
        print(f"Skipping Cross-Validation for {cat_structure} (Iteration {iteration+1}) due to insufficient samples.")

    print(f"Cross-Validation STD: {STD:.4f}")

    results.append({
        'Iteration': iteration + 1,
        'Cat_Structure': cat_structure,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse,
        'CV STD': STD
    })
    print("--------------------------------------")

# Run baseline model for all and individual catalysts
for iteration in range(10):
    train, test = train_test_split(dataset, test_size=0.20, random_state=iteration+47)
    test_all = test[test['Original_Cat_Structure'].isin(original_cat_structure_values)]
    run_experiment(train, test_all, iteration, cat_structure='All')

for cat_structure in original_cat_structure_values:
    for iteration in range(10):
        train, test = train_test_split(dataset, test_size=0.20, random_state=iteration+47)
        test_specific = test[test['Original_Cat_Structure'] == cat_structure]
        if len(test_specific) > 0:
            run_experiment(train, test_specific, iteration, cat_structure=cat_structure)

# Export results
results_df = pd.DataFrame(results)
results_df.to_csv('model_performance_by_cat_structure.csv', index=False)
print("Results exported")

# Compute and export feature importances
avg_feature_importance = {cat_structure: fi / 10 for cat_structure, fi in feature_importance_dict.items()}
feature_importance_df = pd.DataFrame(avg_feature_importance, index=X_names_ycr)
feature_importance_df.to_csv('feature_importance_by_cat_structure.csv', index=True)
print("Feature importance exported")

# ----------------------------------------------
# Select Top 15 Features by 'All' Importance
# ----------------------------------------------
top_n = 15
top_features = feature_importance_df['All'].sort_values(ascending=False).head(top_n).index.tolist()
print(f"Top {top_n} features based on importance: {top_features}")

# Prepare new containers
feature_importance_dict_top = {cat_structure: np.zeros(len(top_features)) for cat_structure in original_cat_structure_values + ['All']}
results_top = []

# Updated experiment runner
def run_experiment_top(train, test, iteration, X_names, cat_structure='All'):
    y_train = train["Yield"]
    y_test = test["Yield"]
    X_train = train[X_names]
    X_test = test[X_names]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    weights = np.where(train['is_synthetic'] == 1, 0.5, 1.0)

    model = RandomForestRegressor(n_estimators=150, min_samples_split=3, min_samples_leaf=1,
                                   max_features='sqrt', max_depth=None, bootstrap=False, random_state=47)

    model.fit(X_train_scaled, y_train, sample_weight=weights)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    train_r2 = np.corrcoef(y_train, y_train_pred)[0, 1]**2
    test_r2 = np.corrcoef(y_test, y_test_pred)[0, 1]**2
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    feature_importance_dict_top[cat_structure] += model.feature_importances_

    print(f"[TOP FEATURES] Iteration {iteration+1} | Cat_Structure: {cat_structure}")
    print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}")

    if len(X_test_scaled) >= 2:
        cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=47)
        n_scores = cross_val_score(model, X_test_scaled, y_test,
                                   scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        std_cv = np.std(n_scores)
    else:
        std_cv = 0
        print(f"Skipping CV for {cat_structure} (Iteration {iteration+1})")

    results_top.append({
        'Iteration': iteration + 1,
        'Cat_Structure': cat_structure,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse,
        'CV STD': std_cv
    })

# Run again using top 15 features
for iteration in range(10):
    train, test = train_test_split(dataset, test_size=0.20, random_state=iteration+47)
    test_all = test[test['Original_Cat_Structure'].isin(original_cat_structure_values)]
    run_experiment_top(train, test_all, iteration, top_features, cat_structure='All')

for cat_structure in original_cat_structure_values:
    for iteration in range(10):
        train, test = train_test_split(dataset, test_size=0.20, random_state=iteration+47)
        test

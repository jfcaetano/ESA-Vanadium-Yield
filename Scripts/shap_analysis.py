#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:45:26 2025

@author: jfcaetano
"""

#### V 5 ESA + SHAP

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import nan_to_num, std
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings('ignore')

# Load and prepare dataset
dataset = pd.read_csv('Augmented_ESA_Vanadium_Database.csv').fillna(0)

# Filter by selected catalyst structures
original_cat_structure_values = ['VO(acac)2', 'VO(OiPr)3', 'VCl2(salen)', 'VO(salen)', 'VOSO4']

# Columns to exclude
exclude_cols = ['Cat_Structure', 'Original_Lab', 'Original_Cat_Structure', 'Laboratory', 'Solvent', 'Cat_Structure', 'Catalyst', 
                'Substrate', 'Ligand', 'Oxidant', 'EE', 'Yield', 'Configuration', 'Cat-SMILES', 
                "Lig-SMILES", "Sub-SMILES", "Sol-SMILES", 'Yield_bin']

# Feature columns
X_names_ycr = [x for x in dataset.columns if x not in exclude_cols]

# Containers for results and feature importances
results = []
feature_importance_dict = {cat_structure: np.zeros(len(X_names_ycr)) for cat_structure in original_cat_structure_values + ['All']}

# Model training and evaluation function
def run_experiment(train, test, iteration, cat_structure='All'):

    y_train_ycr = train.loc[:, "Yield"]
    y_test_ycr = test.loc[:, "Yield"]
    X_train_ycr = train.loc[:, X_names_ycr]
    X_test_ycr = test.loc[:, X_names_ycr]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ycr)
    X_test_scaled = scaler.transform(X_test_ycr)

    weights = np.where(train['is_synthetic'] == 1, 0.5, 1.0)

    model_ycr = RandomForestRegressor(
        n_estimators=150, min_samples_split=3, min_samples_leaf=1,
        max_features='sqrt', max_depth=None, bootstrap=False, random_state=47
    )
    
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

    # Cross-validation
    if len(X_test_scaled) >= 2:
        cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=47)
        n_scores = cross_val_score(model_ycr, X_test_scaled, y_test_ycr, 
                                   scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        STD = np.std(n_scores)
    else:
        STD = 0 
        print(f"Skipping Cross-Validation for {cat_structure} (Iteration {iteration+1}) due to insufficient samples.")

    print(f"Cross-Validation STD: {STD:.4f}")
    print("--------------------------------------")

    results.append({
        'Iteration': iteration + 1,
        'Cat_Structure': cat_structure,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse,
        'CV STD': STD
    })

    if cat_structure == 'All' and iteration == 9:
        return model_ycr, X_test_scaled, y_test_ycr, y_test_ycr.index, scaler
    else:
        return None, None, None, None, None

# Run 10 iterations for all catalysts combined
final_model = None
final_X_test = None
final_y_test = None
final_index = None
final_scaler = None

for iteration in range(10):
    train, test = train_test_split(dataset, test_size=0.20, random_state=iteration+47)
    test_all = test[test['Original_Cat_Structure'].isin(original_cat_structure_values)]
    model_ycr, X_test_scaled, y_test_ycr, y_index, scaler = run_experiment(train, test_all, iteration, cat_structure='All')
    if model_ycr is not None:
        final_model = model_ycr
        final_X_test = X_test_scaled
        final_y_test = y_test_ycr
        final_index = y_index
        final_scaler = scaler

# Run 10 iterations for individual catalysts
for cat_structure in original_cat_structure_values:
    for iteration in range(10):
        train, test = train_test_split(dataset, test_size=0.20, random_state=iteration+47)
        test_specific = test[test['Original_Cat_Structure'] == cat_structure]
        
        if len(test_specific) > 0:
            run_experiment(train, test_specific, iteration, cat_structure=cat_structure)

# Export results
results_df = pd.DataFrame(results)
#results_df.to_csv('model_performance_by_cat_structure.csv', index=False)
print("Results exported")

# Average feature importances
avg_feature_importance = {cat_structure: fi / 10 for cat_structure, fi in feature_importance_dict.items()}
feature_importance_df = pd.DataFrame(avg_feature_importance, index=X_names_ycr)
#feature_importance_df.to_csv('feature_importance_by_cat_structure.csv', index=True)
print("Feature importance exported")

# SHAP Analysis for the 'All' scenario
if final_model is not None:
    explainer = shap.Explainer(final_model, final_X_test)
    shap_values = explainer(final_X_test, check_additivity=False)

    # SHAP summary plot
    shap.summary_plot(shap_values, features=final_X_test, feature_names=X_names_ycr, show=False)
    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Optional: save SHAP values as DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=X_names_ycr, index=final_index)
    shap_df.to_csv("shap_values_all_catalysts.csv")

    print("SHAP analysis for 'All' catalysts scenario completed.")

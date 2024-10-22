#### V 5 ESA

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import random



dataset = pd.read_csv('Augmented_ESA_Vanadium_Database.csv').fillna(0)

# Filter the dataset by `Original_Cat_Structure`
original_cat_structure_values = ['VO(acac)2', 'VO(OiPr)3', 'VCl2(salen)', 'VO(salen)', 'VOSO4']

# Exclude irrelevant columns
exclude_cols = ['Cat_Structure', 'Original_Lab', 'Original_Cat_Structure', 'Laboratory', 'Solvent', 'Cat_Structure', 'Catalyst', 
                'Substrate', 'Ligand', 'Oxidant', 'EE', 'Yield', 'Configuration', 'Cat-SMILES', 
                "Lig-SMILES", "Sub-SMILES", "Sol-SMILES", 'Yield_bin']

# Prepare the feature names
X_names_ycr = [x for x in dataset.columns if x not in exclude_cols]

# Prepare to store results and feature importances
results = []
feature_importance_dict = {cat_structure: np.zeros(len(X_names_ycr)) for cat_structure in original_cat_structure_values + ['All']}

# Function to train and evaluate the model, with feature importance tracking
def run_experiment(train, test, iteration, cat_structure='All'):

    y_train_ycr = train.loc[:, "Yield"]
    y_test_ycr = test.loc[:, "Yield"]
    X_train_ycr = train.loc[:, X_names_ycr]
    X_test_ycr = test.loc[:, X_names_ycr]


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ycr)
    X_test_scaled = scaler.transform(X_test_ycr)

    # Adjust weights for adding less weight to synthetic reactions
    weights = np.where(train['is_synthetic'] == 1, 0.5, 1.0)

    # Initialize model for 'Yield' regression task
    model_ycr = RandomForestRegressor(n_estimators=350, max_depth=40, random_state=47)
    model_ycr.fit(X_train_scaled, y_train_ycr, sample_weight=weights)

    # Evaluate the performance of 'Yield'
    y_train_fitted_ycr = model_ycr.predict(X_train_scaled)
    y_test_fitted_ycr = model_ycr.predict(X_test_scaled)
    train_r2 = np.corrcoef(y_train_ycr, y_train_fitted_ycr)[0, 1]**2
    test_r2 = np.corrcoef(y_test_ycr, y_test_fitted_ycr)[0, 1]**2
    test_mae = mean_absolute_error(y_test_ycr, y_test_fitted_ycr)
    test_rmse = np.sqrt(mean_squared_error(y_test_ycr, y_test_fitted_ycr))

    # Record the results
    results.append({
        'Iteration': iteration + 1,
        'Cat_Structure': cat_structure,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse
    })

    feature_importance_dict[cat_structure] += model_ycr.feature_importances_
    
    print(f"Iteration {iteration+1} | Cat_Structure: {cat_structure} results:")
    print(f"Train 'Yield' R2: {train_r2}")
    print(f"Test 'Yield' R2: {test_r2}")
    print(f"Test MAE: {test_mae}")
    print(f"Test RMSE: {test_rmse}")
    print("--------------------------------------")

# Run 10 iterations for all catalysts combined
for iteration in range(10):
    train, test = train_test_split(dataset, test_size=0.20, random_state=iteration+47)
    test_all = test[test['Original_Cat_Structure'].isin(original_cat_structure_values)]
    run_experiment(train, test_all, iteration, cat_structure='All')

# Run 10 iterations for each specific catalysts individually
for cat_structure in original_cat_structure_values:
    for iteration in range(10):
        train, test = train_test_split(dataset, test_size=0.20, random_state=iteration+47)
        test_specific = test[test['Original_Cat_Structure'] == cat_structure]
        
        if len(test_specific) > 0:
            run_experiment(train, test_specific, iteration, cat_structure=cat_structure)

results_df = pd.DataFrame(results)
results_df.to_csv('model_performance_by_cat_structure.csv', index=False)
print("Results exported")

# Compute average feature importances for each cat_structure
avg_feature_importance = {cat_structure: fi / 10 for cat_structure, fi in feature_importance_dict.items()}
feature_importance_df = pd.DataFrame(avg_feature_importance, index=X_names_ycr)

# Export feature importance
feature_importance_df.to_csv('feature_importance_by_cat_structure.csv', index=True)
print("Feature importance exported")

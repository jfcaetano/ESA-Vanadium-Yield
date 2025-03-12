# WEIGHT ANALYSIS

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np



dataset = pd.read_csv('Augmented_ESA_Vanadium_Database.csv').fillna(0)

synthetic_weights = np.arange(0.1, 1.0, 0.1)

original_cat_structure_values = ['VO(acac)2', 'VO(OiPr)3', 'VCl2(salen)', 'VO(salen)', 'VOSO4']

exclude_cols = ['Cat_Structure', 'Original_Lab', 'Original_Cat_Structure', 'Laboratory', 'Solvent', 'Cat_Structure', 'Catalyst',
                'Substrate', 'Ligand', 'Oxidant', 'EE', 'Yield', 'Configuration', 'Cat-SMILES',
                "Lig-SMILES", "Sub-SMILES", "Sol-SMILES", 'Yield_bin']

X_names_ycr = [x for x in dataset.columns if x not in exclude_cols]

results = []

def run_experiment(train, test, iteration, cat_structure='All', synthetic_weight=0.5):

    y_train_ycr = train.loc[:, "Yield"]
    y_test_ycr = test.loc[:, "Yield"]
    X_train_ycr = train.loc[:, X_names_ycr]
    X_test_ycr = test.loc[:, X_names_ycr]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ycr)
    X_test_scaled = scaler.transform(X_test_ycr)

    weights = np.where(train['is_synthetic'] == 1, synthetic_weight, 1.0)

    model_ycr = GradientBoostingRegressor(n_estimators=150, min_samples_split=4, min_samples_leaf=3, 
                                          max_depth=40, random_state=iteration+47)
    model_ycr.fit(X_train_scaled, y_train_ycr, sample_weight=weights)

    y_train_fitted_ycr = model_ycr.predict(X_train_scaled)
    y_test_fitted_ycr = model_ycr.predict(X_test_scaled)
    train_r2 = np.corrcoef(y_train_ycr, y_train_fitted_ycr)[0, 1]**2
    test_r2 = np.corrcoef(y_test_ycr, y_test_fitted_ycr)[0, 1]**2
    test_mae = mean_absolute_error(y_test_ycr, y_test_fitted_ycr)
    test_rmse = np.sqrt(mean_squared_error(y_test_ycr, y_test_fitted_ycr))

    results.append({
        'Iteration': iteration + 1,
        'Cat_Structure': cat_structure,
        'Synthetic Weight': synthetic_weight,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse
    })

    print(f"Iteration {iteration+1} | Cat_Structure: {cat_structure} | Synthetic Weight: {synthetic_weight} results:")
    print(f"Train 'Yield' R2: {train_r2}")
    print(f"Test 'Yield' R2: {test_r2}")
    print(f"Test MAE: {test_mae}")
    print(f"Test RMSE: {test_rmse}")
    print("--------------------------------------")

# Run experiments for each synthetic weight
for synthetic_weight in synthetic_weights:
    for iteration in range(10):
        train, test = train_test_split(dataset, test_size=0.20, random_state=iteration+47)
        test_all = test[test['Original_Cat_Structure'].isin(original_cat_structure_values)]
        run_experiment(train, test_all, iteration, cat_structure='All', synthetic_weight=synthetic_weight)

results_df = pd.DataFrame(results)
results_df.to_csv('Aug.csv', index=False)
print("Results saved")

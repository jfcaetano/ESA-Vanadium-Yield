import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import randint
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
dataset = pd.read_csv('ESA-Vanadium-Database-with-descriptors.csv').fillna(0)

# Exclude irrelevant columns
exclude_cols = ['Cat_Structure', 'Original_Lab', 'Laboratory', 'Solvent', 'Catalyst', 'Substrate', 'Ligand', 'Oxidant', 'EE', 'Yield', 'Configuration', 'Cat-SMILES', 'Lig-SMILES', 'Sub-SMILES', 'Sol-SMILES', 'Yield_bin']

# Prepare the feature names
X_names_ycr = [x for x in dataset.columns if x not in exclude_cols]

# Prepare to store results
results = []

# Define Random Forest hyperparameter search space
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),
    'bootstrap': [True, False]
}

# Function to run hyperparameter search for RandomForestRegressor
def run_hyperparameter_optimization(X_train, y_train):
    rf = RandomForestRegressor(random_state=47)
    
    # Perform randomized search for hyperparameters
    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=10, scoring='neg_mean_squared_error', 
                                       n_jobs=-1, cv=3, random_state=47)
    
    random_search.fit(X_train, y_train)
    
    # Return the best estimator
    return random_search.best_estimator_, random_search.best_params_

# Function to train and evaluate the Random Forest model with optimized hyperparameters
def run_experiment(train, test, iteration):
    y_train_ycr = train.loc[:, "Yield"]
    y_test_ycr = test.loc[:, "Yield"]
    X_train_ycr = train.loc[:, X_names_ycr]
    X_test_ycr = test.loc[:, X_names_ycr]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ycr)
    X_test_scaled = scaler.transform(X_test_ycr)

    # Perform hyperparameter optimization
    best_rf_model, best_params = run_hyperparameter_optimization(X_train_scaled, y_train_ycr)

    # Train the model with the best hyperparameters
    best_rf_model.fit(X_train_scaled, y_train_ycr)

    # Evaluate the model
    y_train_fitted_ycr = best_rf_model.predict(X_train_scaled)
    y_test_fitted_ycr = best_rf_model.predict(X_test_scaled)
    train_r2 = np.corrcoef(y_train_ycr, y_train_fitted_ycr)[0, 1] ** 2
    test_r2 = np.corrcoef(y_test_ycr, y_test_fitted_ycr)[0, 1] ** 2
    test_mae = mean_absolute_error(y_test_ycr, y_test_fitted_ycr)
    test_rmse = np.sqrt(mean_squared_error(y_test_ycr, y_test_fitted_ycr))

    # Record the results
    results.append({
        'Iteration': iteration + 1,
        'Model': 'Random Forest with Hyperparameter Optimization',
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse,
        'Best Params': best_params
    })

    print(f"Iteration {iteration + 1} | Random Forest Hyperparameter Optimization results:")
    print(f"Best Parameters: {best_params}")
    print(f"Train 'Yield' R²: {train_r2}")
    print(f"Test 'Yield' R²: {test_r2}")
    print(f"Test MAE: {test_mae}")
    print(f"Test RMSE: {test_rmse}")
    print("--------------------------------------")

# Run 10 iterations for the entire dataset (All structures)
for iteration in range(10):
    train, test = train_test_split(dataset, test_size=0.20, random_state=iteration + 47)
    run_experiment(train, test, iteration)

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('random_forest_hyperparameter_optimization_results.csv', index=False)
print("Results exported")

### ESA Yield Prediction
### JFCAETANO 2024
### MIT Licence

import sys, time, csv, math
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn import ensemble
from sklearn import inspection
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import sem

def feature_engineering(df):
    # Example: Creating new features by combining existing features
    df['Lig/Oxi Qt'] = df['Ligand_quant_mmol'] / (df['Oxidant_quant_mmol'] + 1)
    df['Lig/Sub Qt'] = df['Ligand_quant_mmol'] / (df['Substrate_quant_mmol'] + 1)
    df['Oxi/Sub Qt'] = df['Oxidant_quant_mmol'] / (df['Substrate_quant_mmol'] + 1)
    df['Sub/Oxi Qt'] = df['Substrate_quant_mmol'] / (df['Oxidant_quant_mmol'] + 1)
    df['Lig/Cat Qt'] = df['Ligand_quant_mmol'] / (df['Catalyst_quant_mmol'] + 1)
    df['Oxi/Cat Qt'] = df['Oxidant_quant_mmol'] / (df['Catalyst_quant_mmol'] + 1)
    df['Sub/Cat Qt'] = df['Substrate_quant_mmol'] / (df['Catalyst_quant_mmol'] + 1)
    df['Time_h/Temp_K Qt'] = df['Time_h'] / (df['Temp_K']+1)
    
    df['EVSA 1-Lig'] = df['EState_VSA1_Lig'] / (df['VSA_EState1_Lig'] + 1)
    df['EVSA 2-Lig'] = df['EState_VSA2_Lig'] / (df['VSA_EState2_Lig'] + 1)
    df['EVSA 3-Lig'] = df['EState_VSA3_Lig'] / (df['VSA_EState3_Lig'] + 1)
    df['EVSA 4-Lig'] = df['EState_VSA4_Lig'] / (df['VSA_EState4_Lig'] + 1)
    df['EVSA 5-Lig'] = df['EState_VSA5_Lig'] / (df['VSA_EState5_Lig'] + 1)
    df['EVSA 6-Lig'] = df['EState_VSA6_Lig'] / (df['VSA_EState6_Lig'] + 1)
    df['EVSA 7-Lig'] = df['EState_VSA7_Lig'] / (df['VSA_EState7_Lig'] + 1)
    df['EVSA 8-Lig'] = df['EState_VSA8_Lig'] / (df['VSA_EState8_Lig'] + 1)
    df['EVSA 9-Lig'] = df['EState_VSA9_Lig'] / (df['VSA_EState9_Lig'] + 1)
    df['EVSA 10-Lig'] = df['EState_VSA10_Lig'] / (df['VSA_EState10_Lig'] + 1)
   
    # Example: Applying transformations
    df['log Time_h'] = np.log(df['Time_h'] + 1)
    df['log Temp_K'] = np.log(df['Temp_K'] + 1)
    df['log Ligand_quant_mmol'] = np.log(df['Ligand_quant_mmol'] + 1)
    df['log Oxidant_quant_mmol'] = np.log(df['Oxidant_quant_mmol'] + 1)
    df['log Substrate_quant_mmol'] = np.log(df['Substrate_quant_mmol'] + 1)
    df['log Catalyst_quant_mmol'] = np.log(df['Catalyst_quant_mmol'] + 1)

    df = pd.get_dummies(df, columns=['Solvent'])
    df = pd.get_dummies(df, columns=['Oxidant'])
    df = pd.get_dummies(df, columns=['Cat_Structure'])
    df = pd.get_dummies(df, columns=['Ligand'])

    return df


# Function to run the model and return metrics
def run_model():
    # Load the dataset
    df0 = pd.read_csv('Database_1.csv')
    df0 = feature_engineering(df0)

    # Initialize variables
    train_frac = 0.6
    max_group = 16
    train_frames = []
    test_frames = []

    # Loop through each group
    for group in range(1, max_group + 1):
        df_group = df0[df0['Group'] <= group]
        train_group = df_group.sample(frac=train_frac)
        test_group = df_group.drop(train_group.index)

        train_frames.append(train_group)
        test_frames.append(test_group)

        df0 = df0.drop(df_group.index)

    # Concatenate all frames
    ESA_train = pd.concat(train_frames)
    ESA_test = pd.concat(test_frames)

    # Define columns to exclude
    exclude_cols = ['Solvent', 'Cat_Structure', 'Catalyst', 'Substrate', 'Ligand', 'Oxidant', 'Yield', 'EE', 'Configuration', 'Entry']
    X_names = [x for x in df0.columns if x not in exclude_cols]
    Nome=len(X_names)
    
    # Prepare training and testing sets
    y_train = ESA_train.loc[:, "Yield"]
    y_test = ESA_test.loc[:, "Yield"]
    X_train = ESA_train.loc[:, X_names].fillna(0)
    X_test = ESA_test.loc[:, X_names].fillna(0)

    # Fit the model
    model = GradientBoostingRegressor(random_state=47)
    model.fit(X_train, y_train)

    # Make predictions
    y_test_fitted = model.predict(X_test)

    # Calculate residuals and perform IQR outlier detection
    ESA_test['residual'] = y_test - y_test_fitted
    Q1 = ESA_test['residual'].quantile(0.35)
    Q3 = ESA_test['residual'].quantile(0.65)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    ESA_test['outlier'] = (ESA_test['residual'] < lower_bound) | (ESA_test['residual'] > upper_bound)

    # Filter out outliers
    non_outliers = ESA_test[~ESA_test['outlier']]
    y_test_filtered = non_outliers['Yield']
    X_test_filtered = non_outliers[X_names]

    # Handle NaN values in the filtered test set
    X_test_filtered = X_test_filtered.fillna(0)
    
    p_imp = inspection.permutation_importance(model, X_test_filtered, y_test_filtered, n_repeats=5)
    p_imp_av = p_imp['importances_mean']
    
    feature_importance_df = pd.DataFrame({'Feature': X_names, 'Importance': p_imp_av})

    # Identify bad features (very low importance)
    relative_importance = (p_imp_av / sum(p_imp_av))
    bad_features = [name for name, importance in zip(X_names, relative_importance) if importance < 0.00001]

    # Remove bad features from the datasets
    X_train = X_train.drop(columns=bad_features)
    X_test_filtered = X_test_filtered.drop(columns=bad_features)


    X_test_filtered = X_test_filtered.fillna(0)
    X_train = X_train.fillna(0)
    
    # Re-fit the model with updated datasets
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model on the filtered test set
    y_test_fitted_filtered = model.predict(X_test_filtered)
    y_train_fitted = model.predict(X_train)
    Score_train = model.score(X_train, y_train)
    Score_test_filtered = model.score(X_test_filtered, y_test_filtered)
    rsq_train = np.corrcoef(y_train, y_train_fitted)[0, 1] ** 2
    rsq_test = np.corrcoef(y_test_filtered, y_test_fitted_filtered)[0, 1] ** 2
    MSE_filtered = np.square(np.subtract(y_test_filtered, y_test_fitted_filtered)).mean()
    RMSE_filtered = math.sqrt(MSE_filtered)
    cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=47)
    n_scores = cross_val_score(model, X_test_filtered, y_test_filtered, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    STD=std(n_scores)
    MAE=mean_absolute_error(y_test_filtered, y_test_fitted_filtered)
    return MAE, Score_test_filtered, Score_train, rsq_test, rsq_train, RMSE_filtered, STD, len(X_train), len(X_test_filtered), feature_importance_df

# Run the model 10 times and store results
results = []
for i in range(10):
    # Unpack the results, including the feature importance DataFrame
    *result, feature_importance_df = run_model()

    results.append((i + 1, *result))

    # Export feature importance DataFrame to CSV
    feature_importance_df.to_csv(f'feature_importance_{i+1}.csv', index=False)

# Convert results to DataFrame and export to CSV
columns = ['Iteration', 'MAE', 'Score_Test', 'Score_Train', "R2test", "R2train", "RMSE", "STD", 'Train_Size', 'Test_Size']
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv('model_results.csv', index=False)

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


# Load the dataset and fill missing values with 0
data = pd.read_csv('ESA-Vanadium-Database-with-descriptors.csv').fillna(0)

# Define numeric variables for constraints
numeric_variables = ['Temp_K', 'Yield', 'EE', 'Substrate_quant_mmol', 'Catalyst_quant_mmol',
                     'Ligand_quant_mmol', 'Oxidant_quant_mmol', 'Additive_quant_mL',
                     'Solution_vol_mL', 'Time_h']

# Ensure numeric variables are of numeric type
data[numeric_variables] = data[numeric_variables].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=numeric_variables)

# Group data by laboratory and compute min/max constraints
grouped_labs = data.groupby('Laboratory')
lab_constraints_list = []

for lab_name, group in grouped_labs:
    lab_constraints = {'Laboratory': lab_name}
    for variable in numeric_variables:
        lab_constraints[f'{variable}_min'] = group[variable].min()
        lab_constraints[f'{variable}_max'] = group[variable].max()
    
    lab_constraints_list.append(lab_constraints)

# Convert constraints to a DataFrame
lab_constraints = pd.DataFrame(lab_constraints_list)

# Function to apply lab constraints
def apply_lab_constraints(row, constraints_df):
    lab_constraints = constraints_df[constraints_df['Laboratory'] == row['Laboratory']]
    if lab_constraints.empty:
        return False
    lab_constraints = lab_constraints.iloc[0]
    for var in numeric_variables:
        min_val = lab_constraints[f'{var}_min']
        max_val = lab_constraints[f'{var}_max']
        if not (min_val <= row[var] <= max_val):
            return False
    return True

# Filter data using lab constraints
filtered_data = data[data.apply(apply_lab_constraints, axis=1, constraints_df=lab_constraints)]

# Generate synthetic rows with fixed variation
def generate_synthetic_row_with_fixed_variation(lab_name, lab_constraints, numeric_variables, original_row, fixed_variation=0.1):
    synthetic_row = original_row.copy()
    synthetic_row['is_synthetic'] = 1
    lab_constraint = lab_constraints[lab_constraints['Laboratory'] == lab_name].iloc[0]
    
    for var in numeric_variables:
        min_val = lab_constraint[f'{var}_min']
        max_val = lab_constraint[f'{var}_max']
        
        if var in ['Yield', 'EE']:
            min_val = max(0, min_val)
            max_val = min(100, max_val)
        
        current_val = synthetic_row[var]
        variation = random.uniform(0, 1) * fixed_variation * current_val
        
        synthetic_value = min(max(current_val + variation, min_val), max_val)
        synthetic_row[var] = synthetic_value
    
    return synthetic_row

# Add synthetic rows
num_synthetic_rows = 20
fixed_variation = 0.1
synthetic_data = []
for lab_name, lab_group in grouped_labs:
    for _ in range(num_synthetic_rows):
        original_row = lab_group.sample(n=1).iloc[0]
        synthetic_row = generate_synthetic_row_with_fixed_variation(lab_name, lab_constraints, numeric_variables, original_row, fixed_variation=fixed_variation)
        if synthetic_row is not None:
            synthetic_data.append(synthetic_row)

synthetic_data_df = pd.DataFrame(synthetic_data)

# Add 'is_synthetic' column to the original data
filtered_data['is_synthetic'] = 0
combined_data = pd.concat([filtered_data, synthetic_data_df], ignore_index=True)

# Bin the 'Yield' column
combined_data['Yield_bin'] = pd.cut(combined_data['Yield'], bins=10, labels=False, include_lowest=True)

# Determine average number of points per bin
bin_counts = combined_data['Yield_bin'].value_counts().sort_index()
average_count = bin_counts.mean()

# Identify underrepresented bins
threshold = average_count * 0.5
underrepresented_bins = bin_counts[bin_counts < threshold]

# Generate synthetic rows for underrepresented bins
def generate_synthetic_rows_for_underrepresented_bins(data, underrepresented_bins, num_synthetic_rows_per_bin=20, fixed_variation=0.1):
    synthetic_data = []
    
    for bin_num in underrepresented_bins.index:
        bin_data = data[data['Yield_bin'] == bin_num]
        
        if bin_data.empty:
            continue
        
        for _ in range(num_synthetic_rows_per_bin):
            original_row = bin_data.sample(n=1, replace=True).iloc[0]
            lab_name = original_row['Laboratory']
            synthetic_row = generate_synthetic_row_with_fixed_variation(
                lab_name, lab_constraints, numeric_variables, original_row, fixed_variation=fixed_variation)
            
            if synthetic_row is not None:
                synthetic_data.append(synthetic_row)
    
    return pd.DataFrame(synthetic_data)

# Generate synthetic rows for underrepresented bins
synthetic_underrepresented_data = generate_synthetic_rows_for_underrepresented_bins(combined_data, underrepresented_bins, num_synthetic_rows_per_bin=20)

if not synthetic_underrepresented_data.empty:
    combined_data = pd.concat([combined_data, synthetic_underrepresented_data], ignore_index=True)

# One-hot encoding of categorical columns
columns_to_encode = ['Cat_Structure', 'Laboratory', 'Catalyst', 'Substrate', 'Ligand', 'Oxidant', 'Solvent']
combined_data[columns_to_encode] = combined_data[columns_to_encode].astype(str).replace('0', 'missing')
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_columns = one_hot_encoder.fit_transform(combined_data[columns_to_encode])
encoded_df = pd.DataFrame(encoded_columns, columns=one_hot_encoder.get_feature_names_out(columns_to_encode))

# Merge encoded columns and drop original ones
combined_data = combined_data.drop(columns_to_encode, axis=1)
combined_data = pd.concat([combined_data.reset_index(drop=True), encoded_df], axis=1)

# Save the final dataset
combined_data.to_csv('Augmented_ESA_Vanadium_Database.csv', index=False)

### ESA Yield Prediction
### JFCAETANO 2024
### MIT Licence

#MODEL REPORT ANALYSIS

import pandas as pd
import os
from statistics import stdev

# Insert list of CSV file paths
csv_files =['feature_importance_.csv']

# Initialize a dictionary to store total importances and counts


# Initialize a dictionary to store all importances for each feature
all_importances = {}

# Process each file
for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            feature = row['Feature']
            importance = row['Importance']

            if feature in all_importances:
                all_importances[feature].append(importance)
            else:
                all_importances[feature] = [importance]

# Prepare data for DataFrame
data = []
for feature, importances in all_importances.items():
    avg_importance = sum(importances) / len(importances)
    std_dev_importance = stdev(importances) if len(importances) > 1 else 0
    count = len(importances)
    data.append([feature, avg_importance, std_dev_importance, count])

# Convert to DataFrame
features_df = pd.DataFrame(data, columns=['Feature', 'Average Importance', 'Std Deviation', 'Count'])

# Export to new CSV
features_df.to_csv('summary.csv', index=False)


####### GROUP BY DESCRIPTOR FAMILIES

import pandas as pd

# Load the family data
df_family = pd.read_csv('Group-Lists.csv')

# Create lists for each family, excluding NaN values
families = {'Ligand' : [item for item in df_family['Ligand'] if pd.notna(item)],
'Catalyst' : [item for item in df_family['Catalyst'] if pd.notna(item)],
'Substrate' : [item for item in df_family['Substrate'] if pd.notna(item)],
'Solvent' : [item for item in df_family['Solvent'] if pd.notna(item)],
'Reaction' : [item for item in df_family['Reaction'] if pd.notna(item)],
'Engineered' : [item for item in df_family['Engineered'] if pd.notna(item)]}


# Create a mapping of feature to family
feature_to_family = {}
for family, features in families.items():
    for feature in features:
        feature_to_family[feature] = family

# Load the existing features importance DataFrame
df_features = pd.read_csv('summary.csv')

# Map each feature to its family
df_features['Family'] = df_features['Feature'].map(feature_to_family)

# Handle features not found in any family by assigning a default category (optional)
df_features['Family'].fillna('Other', inplace=True)

# Export the updated DataFrame to CSV
df_features.to_csv('updated_list.csv', index=False)

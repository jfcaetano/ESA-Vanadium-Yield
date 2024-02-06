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

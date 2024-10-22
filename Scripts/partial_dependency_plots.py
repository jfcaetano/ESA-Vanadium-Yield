################################

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D  # Import Line2D for custom legend handles

# Set a global style using seaborn
sns.set(style="whitegrid", palette="Paired")

# Load your dataset
dataset = pd.read_csv('Augmented_ESA_Vanadium_Database.csv')

# Filter the dataset by `Original_Cat_Structure`
exclude_cols = ['Cat_Structure', 'Original_Lab', 'Original_Cat_Structure', 'Laboratory', 'Solvent', 'Cat_Structure', 'Catalyst', 
                'Substrate', 'Ligand', 'Oxidant', 'EE', 'Yield', 'Configuration', 'Cat-SMILES', 
                "Lig-SMILES", "Sub-SMILES", "Sol-SMILES", 'Yield_bin']

# Prepare the feature names
X_names_ycr = [x for x in dataset.columns if x not in exclude_cols]

# Split your data into training and testing sets
train = dataset.sample(frac=0.8, random_state=47)
test = dataset.drop(train.index)

# Prepare the data for the model
y_train_ycr = train.loc[:, "Yield"]
X_train_ycr = train.loc[:, X_names_ycr]

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ycr)

# Adjust weights for synthetic reactions
weights = np.where(train['is_synthetic'] == 1, 0.5, 1.0)

# Initialize and train the RandomForestRegressor
model_ycr = RandomForestRegressor(n_estimators=350, max_depth=40, random_state=47)
model_ycr.fit(X_train_scaled, y_train_ycr, sample_weight=weights)

# PARTIAL DEPENDENCE PLOTS

# Select the features for Partial Dependence Plots
features_to_plot = ['Catalyst_quant_mmol', 'Ligand_quant_mmol', 'Oxidant_quant_mmol']

# Store the partial dependence data for each feature
pdp_data = {}

# Generate Partial Dependence Plots and extract data
for feature in features_to_plot:
    feature_index = X_names_ycr.index(feature)
    
    pdp_display = PartialDependenceDisplay.from_estimator(
        model_ycr, 
        X_train_scaled, 
        features=[feature_index], 
        feature_names=X_names_ycr,
        line_kw={'linewidth': 2.5}  # Thicken the lines
    )

    # Extract the PDP data for this feature
    for pd_line in pdp_display.lines_[0]:  # Access the line for this feature
        x_values = pd_line.get_xdata()
        y_values = pd_line.get_ydata()

        # Manually convert the scaled values back to the real scale using the scaler's mean and scale for this feature
        mean = scaler.mean_[feature_index]
        scale = scaler.scale_[feature_index]
        real_x_values = (x_values * scale) + mean

        pdp_data[feature] = {'x_values': real_x_values, 'pdp_values': y_values}

# Define custom colors for each Catalyst Structure
color_map = {
    'VCl2(salen)': 'slategrey',
    'VO(OiPr)3': 'dodgerblue',
    'VO(acac)2': 'limegreen',
    'VO(salen)': 'gold',
    'VOSO4': 'orangered'
}

# Map the Catalyst Structure column to the colors
colors = train['Original_Cat_Structure'].map(color_map)

# Create a combined plot with Yield on the primary y-axis and Partial Dependence on the secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 7))

# Scatter plot of Yield vs. Temp_K, colored by Original_Cat_Structure with increased dot size and black edges
scatter = ax1.scatter(train['Catalyst_quant_mmol'], train['Yield'], c=colors, edgecolor='black', s=100, label='Yield (Catalyst Quantity / mmol)', alpha=0.9)
ax1.set_xlabel('Catalyst quantity / mmol', fontsize=16, color='black')  # Custom X axis title
ax1.set_ylabel('Reaction Yield / %', color='black', fontsize=16)  # Custom Y axis title
ax1.tick_params(axis='both', which='major', labelsize=16, colors='black')
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.spines['right'].set_visible(True)  # Add right spine back
ax1.spines['right'].set_color('black')
ax1.spines['top'].set_color('white')
ax1.set_xlim([-0.001, 0.074])


# Create a secondary y-axis for Partial Dependence
ax2 = ax1.twinx()

# Plot the Partial Dependence line for Temp_K on the secondary y-axis
line_handle, = ax2.plot(pdp_data['Catalyst_quant_mmol']['x_values'], pdp_data['Catalyst_quant_mmol']['pdp_values'], color='black', 
                        label='Partial Dependence (Temp_K)', linewidth=3, alpha=0.5)

# Add error bands (mock error) around the PDP line (for illustration)
error = 0.02 * np.abs(pdp_data['Catalyst_quant_mmol']['pdp_values'])  # Assume 5% error for illustration
ax2.fill_between(pdp_data['Catalyst_quant_mmol']['x_values'], pdp_data['Catalyst_quant_mmol']['pdp_values'] - error,
                 pdp_data['Catalyst_quant_mmol']['pdp_values'] + error, color='dimgrey', alpha=0.1)

ax2.set_ylabel('Partial Dependence', color='black', fontsize=16)
ax2.tick_params(axis='y', labelsize=16, colors='black')
ax2.spines['bottom'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.spines['right'].set_color('black')
ax2.spines['top'].set_color('white')
ax2.set_xlim([-0.001, 0.074])

# Remove the grid
ax1.grid(False)

# Manually create legend handles for each Catalyst Structure
legend_handles = [Line2D([0], [0], marker='o', color='w', label='VCl$_2$(salen)', markerfacecolor='slategrey', markersize=13, markeredgecolor='black'),
                  Line2D([0], [0], marker='o', color='w', label='VO(O$i$Pr)$_3$', markerfacecolor='dodgerblue', markersize=13, markeredgecolor='black'),
                  Line2D([0], [0], marker='o', color='w', label='VO(acac)$_2$', markerfacecolor='limegreen', markersize=13, markeredgecolor='black'),
                  Line2D([0], [0], marker='o', color='w', label='VO(salen)', markerfacecolor='gold', markersize=13, markeredgecolor='black'),
                  Line2D([0], [0], marker='o', color='w', label='VOSO$_4$', markerfacecolor='orangered', markersize=13, markeredgecolor='black')]

# Add the Partial Dependence Line handle to the legend
legend_handles.append(line_handle)

# Combine labels for the custom legend
legend_labels = [r'VCl$_2$(salen)', r'VO(O$i$Pr)$_3$', r'VO(acac)$_2$', r'VO(salen)', r'VOSO$_4$', 'Partial Dependence (Catalyst Quantity / mmol)']

# Create the combined legend
legend = ax1.legend(handles=legend_handles, labels=legend_labels, title='Catalyst Structure and PDP', bbox_to_anchor=(0.5, -0.15), 
                    loc='upper center', ncol=3, fontsize=14, title_fontsize=14, frameon=False)
legend.get_title().set_color('black')

# Add a title and show the plot
plt.title('Yield and Partial Dependence vs. Catalyst Quantity / mmol', fontsize=18, weight='bold')
plt.grid(False)
plt.tight_layout()
plt.savefig('pdp-test-perf-cat-q.png', dpi=900)

plt.show()

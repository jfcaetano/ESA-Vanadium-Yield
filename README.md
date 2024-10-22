# ESA-Vanadium-Yield
Repository for the article "Optimizing Vanadium-Catalyzed ESA Reactions: Machine Learning-Driven Yield Predictions and Data Augmentation"

This repository supports robust machine learning model optimization and interpretability for predicting reaction outcomes in epoxidation processes.

# ESA Model Optimization Project

This repository contains Python scripts used for optimizing machine learning models in the ESA reaction dataset. The scripts are designed for various stages of model training, hyperparameter tuning, data augmentation, and result evaluation. Below is an explanation of the key scripts and their roles.

## Scripts Overview

### 1. `mol-conversion.py`
This script is designed for **molecular data processing and conversion**. It helps convert molecular structures in the dataset into formats that can be used by the machine learning models. Key functionalities include:
- Converting molecular data (e.g., SMILES strings) into descriptor formats suitable for model input.
- Handling molecule-based transformations, ensuring that molecular representations are standardized and compatible with the model.
- The output of this script is typically used as input features for subsequent model training and evaluation.

### 2. `model_preliminary_run.py`
This script performs an **initial model evaluation** to provide baseline performance metrics. It runs a set of pre-defined machine learning models without hyperparameter tuning, giving insight into how well each model performs on the original dataset before further optimization steps. Functions include:
- Loading the dataset and preparing it for modeling.
- Running baseline models: Random Forest, Gradient Boosting, Support Vector Machines, Neural Networks.
- Outputting preliminary performance metrics (`R²`, MAE, RMSE) to help guide future model selection and optimization steps.

### 3. `ESA_data_augmentation.py`
This script is used for **data augmentation** on the ESA-Vanadium dataset. It enriches the dataset by generating synthetic data points or applying transformations to existing data. The key functions of this script include:
- Handling missing values or filling gaps in the dataset.
- Augmenting the dataset to expand the available feature space, enabling more robust model training.
- Ensures that augmented data remains representative of the original chemical reaction conditions, focusing on maintaining chemical consistency.
  
### 4. `ESA-hyperparameter-opt.py`
This script performs **hyperparameter optimization** for a Random Forest model using the `RandomizedSearchCV` function from `scikit-learn`. The primary objective is to optimize the model's performance by searching over a predefined range of hyperparameters. The script:
- Loads the ESA-Vanadium dataset, excluding certain irrelevant features.
- Conducts Random Forest regression, applying a hyperparameter search over parameters like the number of estimators, depth, and minimum samples split.
- Evaluates model performance using metrics such as `R²`, mean absolute error (MAE), and root mean squared error (RMSE) over multiple iterations.
- Saves the optimized model results and the best hyperparameters to a CSV file for further analysis.

### 5. `ESA_model_opt_run.py`
This script focuses on the **execution and evaluation** of the optimized RF machine learning model on the augmented ESA-Vanadium dataset. It is responsible for:
- Collecting results from each model's training and testing phases.
- Recording performance metrics, including training and test `R²`, MAE, and RMSE for each model and iteration.
- Saving the evaluation metrics and model-specific results for further analysis or comparison.

### 7. `partial_dependency_plots.py`
This script generates **partial dependence plots (PDPs)** to interpret the behavior of the trained models. PDPs show how each feature impacts the predicted outcome, allowing for better understanding of the model's decision-making process. The script:
- Loads the trained model and dataset.
- Generates plots that display the relationship between individual features and the model's predictions.
- Visualizes feature importance and dependencies, helping to uncover key drivers in the reaction outcomes.

---


Database/

ESA-Vanadium-Database-v1.csv: Complete dataset with descriptor calculations

Group-Lists.csv: Descriptor group lists

Scripts/

mol-conversion.py: Calculation of desired RdKit descriptors using the raw database

model_initial_run.py: Model calculations using desired algorithms with all calculated descriptors

model_report_analysis.py: Model performance and report analysis

model_feature_eng_run.py: Calculation of engineered features


Results/

ESA-Model-Full-Results.xlsx: File including all model results presented in the paper (including permuation importance, model statistical performance and descriptors group performance determinations)

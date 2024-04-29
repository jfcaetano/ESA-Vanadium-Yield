# ESA-Vanadium-Yield
Repository for the article "Feature Engineered, Machine Learning Models for Explainable Yield Prediction in Vanadium-Catalyzed Epoxidation of Small Alkenes and Allylic Alcohols"

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

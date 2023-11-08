### ESA Yield Prediction
### JFCAETANO 2023
### MIT Licence

import csv, math, sys
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn import ensemble
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from scipy.stats import sem


#Select file
df0 = pd.read_csv('231027-ESA-Vanadium-Database.csv')

#select loads
my_train = [0.60]
my_model = ["GradientBoostingRegressor(n_estimators=250, min_samples_split=2, min_samples_leaf= 1, max_depth=20,random_state=47)"]

#Enumerate groups

Sol=['Chi0v_Sol','MolMR_Sol','SlogP_VSA4_Sol','MaxEStateIndex_Sol','EState_VSA2_Sol']
Cat=['VSA_EState4_Cat','MinEStateIndex_Cat','TPSA_Cat','SMR_VSA1_Cat','MinAbsEStateIndex_Cat','SlogP_VSA3_Cat','Chi1_Cat','NumHAcceptors_Cat','Chi3n_Cat','NumAliphaticRings_Cat','VSA_EState10_Cat','SlogP_VSA2_Cat','NumHeteroatoms_Cat','SlogP_VSA4_Cat']
Lig=['EState_VSA7_Lig','BalabanJ_Lig','VSA_EState4_Lig','SlogP_VSA5_Lig','EState_VSA10_Lig','VSA_EState2_Lig','PEOE_VSA3_Lig','EState_VSA5_Lig','Chi3n_Lig','Kappa2_Lig','PEOE_VSA8_Lig','Chi4v_Lig','SMR_VSA5_Lig','MolLogP_Lig','MaxPartialCharge_Lig','TPSA_Lig','VSA_EState7_Lig']
Sub=['PEOE_VSA9_Sub','EState_VSA10_Sub','VSA_EState8_Sub','Chi2n_Sub','VSA_EState7_Sub','MaxPartialCharge_Sub','MaxEStateIndex_Sub','SMR_VSA6_Sub','MaxAbsEStateIndex_Sub','SlogP_VSA5_Sub','EState_VSA9_Sub','Chi3v_Sub','MinAbsPartialCharge_Sub','MinAbsEStateIndex_Sub','MinPartialCharge_Sub','VSA_EState3_Sub','SlogP_VSA11_Sub','PEOE_VSA6_Sub']
Exp=['Time_h','Oxidant_quant_mmol','Catalyst_quant_mmol','Ligand_quant_mmol','Solvent_Group','Temp_K']

#Group II

# VSA=['PEOE_VSA9_Sub','EState_VSA10_Sub','VSA_EState8_Sub','VSA_EState7_Sub','SMR_VSA6_Sub','SlogP_VSA5_Sub','EState_VSA9_Sub','VSA_EState3_Sub','SlogP_VSA11_Sub','PEOE_VSA6_Sub','VSA_EState4_Cat','TPSA_Cat','SMR_VSA1_Cat','SlogP_VSA3_Cat','VSA_EState10_Cat','SlogP_VSA2_Cat','SlogP_VSA4_Cat','SlogP_VSA4_Sol','EState_VSA2_Sol','EState_VSA7_Lig','VSA_EState4_Lig','SlogP_VSA5_Lig','EState_VSA10_Lig','VSA_EState2_Lig','PEOE_VSA3_Lig','EState_VSA5_Lig','PEOE_VSA8_Lig','SMR_VSA5_Lig','TPSA_Lig','VSA_EState7_Lig']
# Ele=['MaxPartialCharge_Sub','MaxEStateIndex_Sub','MaxAbsEStateIndex_Sub','MinAbsPartialCharge_Sub','MinAbsEStateIndex_Sub','MinPartialCharge_Sub','MinEStateIndex_Cat','MinAbsEStateIndex_Cat','MolMR_Sol','MaxEStateIndex_Sol','MaxPartialCharge_Lig']
# Str=['Chi2n_Sub','Chi3v_Sub','Chi1_Cat','NumHAcceptors_Cat','Chi3n_Cat','NumAliphaticRings_Cat','NumHeteroatoms_Cat','Chi0v_Sol','BalabanJ_Lig','Chi3n_Lig','Kappa2_Lig','Chi4v_Lig','MolLogP_Lig']
# Exp=['Time_h','Oxidant_quant_mmol','Catalyst_quant_mmol','Ligand_quant_mmol','Solvent_Group','Temp_K']


desc=[Sol,Cat,Lig,Sub,Exp]
o = list()
#munber of trials
my_list = [1,2,3,4,5]
for mod in my_model:
    for time in my_list:
        for num in my_train:
            
            train = eval(f"{num}")
            #distribute per bibliographical sources
            df1 = df0[df0['Group'] <= 1]
            train_1 = df1.sample(frac = train)
            test_1 = df1.drop(train_1.index)
            dfx = df0.drop(train_1.index)
            df0 = dfx.drop(test_1.index)

            df2 = df0[df0['Group'] <= 2]
            train_2 = df2.sample(frac = train)
            test_2 = df2.drop(train_2.index)
            dfx = df0.drop(train_2.index)
            df0 = dfx.drop(test_2.index)

            df3 = df0[df0['Group'] <= 3]
            train_3 = df3.sample(frac = train)
            test_3 = df3.drop(train_3.index)
            dfx = df0.drop(train_3.index)
            df0 = dfx.drop(test_3.index)

            df4 = df0[df0['Group'] <= 4]
            train_4 = df4.sample(frac = train)
            test_4 = df4.drop(train_4.index)
            dfx = df0.drop(train_4.index)
            df0 = dfx.drop(test_4.index)

            df5 = df0[df0['Group'] <= 5]
            train_5 = df5.sample(frac = train)
            test_5 = df5.drop(train_5.index)
            dfx = df0.drop(train_5.index)
            df0 = dfx.drop(test_5.index)

            df6 = df0[df0['Group'] <= 6]
            train_6 = df6.sample(frac = train)
            test_6 = df6.drop(train_6.index)
            dfx = df0.drop(train_6.index)
            df0 = dfx.drop(test_6.index)

            df7 = df0[df0['Group'] <= 7]
            train_7 = df7.sample(frac = train)
            test_7 = df7.drop(train_7.index)
            dfx = df0.drop(train_7.index)
            df0 = dfx.drop(test_7.index)

            df8 = df0[df0['Group'] <= 8]
            train_8 = df8.sample(frac = train)
            test_8 = df8.drop(train_8.index)
            dfx = df0.drop(train_8.index)
            df0 = dfx.drop(test_8.index)

            df9 = df0[df0['Group'] <= 9]
            train_9 = df9.sample(frac = train)
            test_9 = df9.drop(train_9.index)
            dfx = df0.drop(train_9.index)
            df0 = dfx.drop(test_9.index)

            df10 = df0[df0['Group'] <= 10]
            train_10 = df10.sample(frac = train)
            test_10 = df10.drop(train_10.index)
            dfx = df0.drop(train_10.index)
            df0 = dfx.drop(test_10.index)

            df11 = df0[df0['Group'] <= 11]
            train_11 = df11.sample(frac = train)
            test_11 = df11.drop(train_11.index)
            dfx = df0.drop(train_11.index)
            df0 = dfx.drop(test_11.index)

            df12 = df0[df0['Group'] <= 12]
            train_12 = df12.sample(frac = train)
            test_12 = df12.drop(train_12.index)
            dfx = df0.drop(train_12.index)
            df0 = dfx.drop(test_12.index)

            df13 = df0[df0['Group'] <= 13]
            train_13 = df13.sample(frac = train)
            test_13 = df13.drop(train_13.index)
            dfx = df0.drop(train_13.index)
            df0 = dfx.drop(test_13.index)

            df14 = df0[df0['Group'] <= 14]
            train_14 = df14.sample(frac = train)
            test_14 = df14.drop(train_14.index)
            dfx = df0.drop(train_14.index)
            df0 = dfx.drop(test_14.index)

            df15 = df0[df0['Group'] <= 15]
            train_15 = df15.sample(frac = train)
            test_15 = df15.drop(train_15.index)
            dfx = df0.drop(train_15.index)
            df0 = dfx.drop(test_15.index)

            df16 = df0[df0['Group'] <= 16]
            train_16 = df16.sample(frac = train)
            test_16 = df16.drop(train_16.index)

            frames0 = [train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9, train_10, train_11, train_12, train_13, train_14, train_15, train_16]
            frames1 = [test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_10, test_11, test_12, test_13, test_14, test_15, test_16]
            ESA_train = pd.concat(frames0)
            ESA_test = pd.concat(frames1)

            for item in desc:
                
                X_names=[x for x in df0.columns if x in eval(f"{item}")]

                y_train=ESA_train.loc[:,"Yield"]
                y_test=ESA_test.loc[:,"Yield"]

                X_train = ESA_train.loc[:,X_names]
                X_test =ESA_test.loc[:,X_names]

                X_train=X_train.fillna(0)
                X_test=X_test.fillna(0)
                
                model = eval(f"{mod}")
                model.fit(X_train, y_train)
                y_train_fitted=model.predict(X_train)
                y_test_fitted=model.predict(X_test)
                rsq_train = np.corrcoef(y_train,y_train_fitted)[0,1]**2
                rsq_test = np.corrcoef(y_test,y_test_fitted)[0,1]**2
                Score_train = model.score(X_train, y_train)
                Score_test = model.score(X_test, y_test)
                MPE=mean_absolute_percentage_error(y_test,y_test_fitted)
                MAE=mean_absolute_error(y_test,y_test_fitted)
                MUE=sem(y_test)*sem(y_test)+sem(y_test_fitted)*sem(y_test_fitted)
                cv = RepeatedKFold(n_splits=2, n_repeats=5, random_state=47)
                n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
                STD=std(n_scores)
                AARD=(100/len(X_test))*(sum(abs((y_test_fitted-y_test)/y_test_fitted)))
                MSE = np.square(np.subtract(y_test,y_test_fitted)).mean() 
                RMSE = math.sqrt(MSE)

                z = {'y_test': y_test, 'y_test_pred': y_test_fitted}
                df = pd.DataFrame(z)
                frames = [df, ESA_test]
                result = pd.concat(frames, axis="columns")
                a=result.y_test
                b=result.y_test_pred
                ev =((a-b)**2)/(a**2)*100
                ev = pd.DataFrame(ev)
                ev.rename( columns={0 :'eval'}, inplace=True)

                frames = [result, ev]
                ev.colums = ['eval']
                full = pd.concat(frames, axis="columns")

                #confidence interval 96%
                full.drop(full[full['eval'] >= 4].index, inplace = True)

                y_test = full['y_test'].to_numpy()
                y_test_pred = full['y_test_pred'].to_numpy()

                X_names=[x for x in df0.columns if x in eval(f"{item}")]
                X_test = full.loc[:,X_names]
                X_test=X_test.fillna(0)


                y_train=np.nan_to_num(y_train, nan=0)
                y_test=np.nan_to_num(y_test, nan=0)

                model.fit(X_train, y_train)
                y_train_fitted=model.predict(X_train)
                y_test_fitted=model.predict(X_test)
                rsq_train = np.corrcoef(y_train,y_train_fitted)[0,1]**2
                rsq_test = np.corrcoef(y_test,y_test_fitted)[0,1]**2
                Score_train = model.score(X_train, y_train)
                Score_test = model.score(X_test, y_test)
                MPE=mean_absolute_percentage_error(y_test,y_test_fitted)
                MAE=mean_absolute_error(y_test,y_test_fitted)
                MUE=sem(y_test)*sem(y_test)+sem(y_test_fitted)*sem(y_test_fitted)
                cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=47)
                n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
                STD=std(n_scores)
                AARD=(100/len(X_test))*(sum(abs((y_test_fitted-y_test)/y_test_fitted)))
                MSE = np.square(np.subtract(y_test,y_test_fitted)).mean() 
                RMSE = math.sqrt(MSE)

                nl = dict()
                nl[f"Algorithm"]=f"{mod}"
                nl[f"Training_Load"]=eval(f"{num}")
                nl[f"Desc_Group"]=eval(f"{item}")
                nl[f"Score_test"]=Score_test
                nl[f"Score_train"]=Score_train
                nl[f"MAE"]=MAE
                nl[f"MPE"]=MPE*100
                nl[f"RMSE"]=RMSE
                nl[f"STD"]=STD
                nl[f"AARD"]=AARD
                nl[f"R2"]=rsq_test
                nl[f"Num"]=len(X_test)

                o.append(nl)

output_fn = 'model-desc_eval-optimized.csv'
with open(output_fn,'w',newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=o[0].keys())
    writer.writeheader()
    for new_row in o:
        writer.writerow(new_row)
############

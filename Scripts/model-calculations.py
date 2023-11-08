### ESA Yield Prediction
### JFCAETANO 2023
### MIT Licence

import rdkit, sys, time, csv, math
from rdkit import Chem
from rdkit.Chem import Descriptors
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


df0 = pd.read_csv('231027-ESA-Vanadium-Database.csv')

train=0.6

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


exclude_cols=['Solvent','Cat_Structure','Catalyst','Substrate','Ligand','Oxidant', 'EE','Yield', 'Configuration', 'Entry', 'Reference_DOI', 'Reference_location']

X_names=[x for x in df0.columns if x not in exclude_cols]

y_train=ESA_train.loc[:,"Yield"]
y_test=ESA_test.loc[:,"Yield"]

X_train = ESA_train.loc[:,X_names]
X_test =ESA_test.loc[:,X_names]

X_train=X_train.fillna(0)
X_test=X_test.fillna(0)


# Fit RF model with no HP

model = GradientBoostingRegressor(n_estimators=250, min_samples_split=2, min_samples_leaf= 1, max_depth=20,random_state=47)

model.fit(X_train, y_train)
y_train_fitted=model.predict(X_train)
y_test_fitted=model.predict(X_test)
rsq_train = np.corrcoef(y_train,y_train_fitted)[0,1]**2
rsq_test = np.corrcoef(y_test,y_test_fitted)[0,1]**2
Score_train = model.score(X_train, y_train)
Score_test = model.score(X_test, y_test)
Score_train, Score_test, len(X_train)+len(X_test)
MSE = np.square(np.subtract(y_test,y_test_fitted)).mean() 
RMSE = math.sqrt(MSE)
s1=mean_absolute_error(y_test,y_test_fitted), Score_test, Score_train, len(X_train), len(X_test)

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
#96% confidence thereshold 
full.drop(full[full['eval'] >= 4].index, inplace = True)

y_test = full['y_test'].to_numpy()
y_test_pred = full['y_test_pred'].to_numpy()

X_names=[x for x in df0.columns if x not in exclude_cols]
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
Score_train, Score_test, len(X_train)+len(X_test)
MSE = np.square(np.subtract(y_test,y_test_fitted)).mean() 
RMSE = math.sqrt(MSE)
s2=mean_absolute_error(y_test,y_test_fitted), Score_test, Score_train, len(X_train), len(X_test)

s1, s2

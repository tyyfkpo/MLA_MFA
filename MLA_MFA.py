# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:30:35 2019

@author: PanQ
"""

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import os
import os.path
import matplotlib.pyplot as plt
#import datetime

from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import warnings  
warnings.filterwarnings("ignore")

truevalue = pd.read_excel(r'NOM_Truevalue.xlsx').fillna(0)
truevalue['ppm'] = abs(truevalue['ppm'])
truevalue = truevalue[truevalue['ppm'] <= 2]
truevalue = truevalue[truevalue['s/n'] >= 6]
totalvalue = pd.read_excel(r'NOM_total_allmatch.xlsx').fillna(0)
totalvalue['ppm'] = abs(totalvalue['ppm'])
totalvalue = totalvalue[totalvalue['ppm'] <= 2]
totalvalue = totalvalue[totalvalue['s/n'] >= 6]



truevalue = truevalue.round({'theo': 6})
totalvalue = totalvalue.round({'theo': 6})
#truevalue.describe()
#truevalue['m/z'] = truevalue['m/z'].round(decimals = 5)
truevaluemzlist = list(truevalue['theo'])
#totalvalue.loc[(totalvalue['m/z'] in truevaluemzlist), 'label'] = 1
index_list = []
for i in totalvalue.itertuples():
    """if i[0] < 3:
        print(i)
    #if 
    """
    theo_mz = i[6]
    if theo_mz in truevaluemzlist:
        #print('OK')
        index_list.append(i[0])   
#print(len(index_list))

totalvalue.loc[index_list,'lable'] = 1
totalvalue['lable'] = totalvalue['lable'].fillna(0)
totalvalue0 = totalvalue


falsevalue = totalvalue[totalvalue['lable'] == 0]
truevalue = totalvalue[totalvalue['lable'] == 1]

totalvalue = totalvalue[['C', 'H', 'N', 'O', 'S', 'theo', 'intensity', 's/n', 'ppm', 'DBE', 'KMD', 'lable']]
totalvalue['ID'] = totalvalue.index
totalvalue[['C', 'H', 'N', 'O', 'S', 'DBE', 'lable','ID']].astype(int)
#falsevalue.describe().to_excel(r'falsevalue_describe.xlsx')
#truevalue.describe().to_excel(r'truevalue_describe.xlsx')

Fdes = falsevalue.describe()
Fdes['ID'] = 'F'
Tdes = truevalue.describe()
Tdes['ID'] = 'T'

Fdes = Fdes.append(Tdes)
Fdes.to_excel(r'NOM_TnF_describe.xlsx')

#==============================================================================
# list(data_test.index)
# [list(totalvalue.index).remove(i) for i in list(data_test.index)]
# 
# data_train = totalvalue.loc[i for i not in data_test.index,:]
#s1.loc[s1.index.difference(s2.index)]
#==============================================================================

data_test = totalvalue.sample(frac=0.90,axis=0)
###print(data_test.head())
data_train = totalvalue.loc[totalvalue.index.difference(data_test.index)]

dummies_N = pd.get_dummies(data_train['N'], prefix= 'N')
dummies_S = pd.get_dummies(data_train['S'], prefix= 'S')

data_train = pd.concat([data_train, dummies_N,dummies_S], axis=1)

data_train.drop(['N','S'],axis = 1, inplace = True)



#scale
#==============================================================================
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

C_scale_param = scaler.fit(data_train['C'])
data_train['C_scaled'] = scaler.fit_transform(data_train['C'], C_scale_param)

H_scale_param = scaler.fit(data_train['H'])
data_train['H_scaled'] = scaler.fit_transform(data_train['H'], H_scale_param)

O_scale_param = scaler.fit(data_train['O'])
data_train['O_scaled'] = scaler.fit_transform(data_train['O'], O_scale_param)

theo_scale_param = scaler.fit(data_train['theo'])
data_train['theo_scaled'] = scaler.fit_transform(data_train['theo'], theo_scale_param)

intensity_scale_param = scaler.fit(data_train['intensity'])
data_train['intensity_scaled'] = scaler.fit_transform(data_train['intensity'], intensity_scale_param)

sn_scale_param = scaler.fit(data_train['s/n'])
data_train['s/n_scaled'] = scaler.fit_transform(data_train['s/n'], sn_scale_param)

ppm_scale_param = scaler.fit(data_train['ppm'])
data_train['ppm_scaled'] = scaler.fit_transform(data_train['ppm'], ppm_scale_param)

DBE_scale_param = scaler.fit(data_train['DBE'])
data_train['DBE_scaled'] = scaler.fit_transform(data_train['DBE'], DBE_scale_param)

KMD_scale_param = scaler.fit(data_train['KMD'])
data_train['KMD_scaled'] = scaler.fit_transform(data_train['KMD'], KMD_scale_param)
#==============================================================================



from sklearn import linear_model


# Use regex to get the attribute value we wantC|H|O|theo|intensity|ppm|DBE|KMD|lable|s/n|
data_train = data_train.filter(regex='lable|C_.*|H_.*|O_.*|theo_.*|intensity_.*|s/n_.*|ppm_.*|DBE_.*|KMD_.*|N_.*|S_.*')
train_np = data_train.as_matrix()
 

y = train_np[:, 0]
 
# X the attributes
X = train_np[:, 1:]
 
# LogisticRegression
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)



####testing set
dummies_N2 = pd.get_dummies(data_test['N'], prefix= 'N')
dummies_S2 = pd.get_dummies(data_test['S'], prefix= 'S')

df_test = pd.concat([data_test, dummies_N2,dummies_S2], axis=1)

df_test.drop(['N','S','lable'],axis = 1, inplace = True)

#scale
#==============================================================================
scaler = preprocessing.StandardScaler()
C_scale_param = scaler.fit(df_test['C'])
df_test['C_scaled'] = scaler.fit_transform(df_test['C'], C_scale_param)

H_scale_param = scaler.fit(df_test['H'])
df_test['H_scaled'] = scaler.fit_transform(df_test['H'], H_scale_param)

O_scale_param = scaler.fit(df_test['O'])
df_test['O_scaled'] = scaler.fit_transform(df_test['O'], O_scale_param)

theo_scale_param = scaler.fit(df_test['theo'])
df_test['theo_scaled'] = scaler.fit_transform(df_test['theo'], theo_scale_param)

intensity_scale_param = scaler.fit(df_test['intensity'])
df_test['intensity_scaled'] = scaler.fit_transform(df_test['intensity'], intensity_scale_param)

sn_scale_param = scaler.fit(df_test['s/n'])
df_test['s/n_scaled'] = scaler.fit_transform(df_test['s/n'], sn_scale_param)

ppm_scale_param = scaler.fit(df_test['ppm'])
df_test['ppm_scaled'] = scaler.fit_transform(df_test['ppm'], ppm_scale_param)

DBE_scale_param = scaler.fit(df_test['DBE'])
df_test['DBE_scaled'] = scaler.fit_transform(df_test['DBE'], DBE_scale_param)

KMD_scale_param = scaler.fit(df_test['KMD'])
df_test['KMD_scaled'] = scaler.fit_transform(df_test['KMD'], KMD_scale_param)
#==============================================================================




test = df_test.filter(regex='C_.*|H_.*|O_.*|theo_.*|intensity_.*|s/n_.*|ppm_.*|DBE_.*|KMD_.*|N_.*|S_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'ID':data_test['ID'].as_matrix(), 'lable':predictions.astype(np.int32), 'lable_manual': data_test['lable'].as_matrix()})

print(pd.DataFrame({"columns":list(data_train.columns)[1:], "coef":list(clf.coef_.T)}))

lbe_totalvalue = totalvalue0

predict_lable = []
for ltrow in lbe_totalvalue.itertuples(): 
    ltrow_idx = ltrow[0]
    predict_lable.append(np.nan)
    
    for rrow in  result.itertuples():
    #if i[0] <2:
        #print(i[1])
        rr_idx = rrow[1]
        rr_lbe = rrow[2]
        #predict_lable.append()
        if ltrow_idx == rr_idx:
            predict_lable[-1] = rr_lbe
            
lbe_totalvalue['predict_lbl'] = predict_lable    
lbe_totalvalue['type'] = lbe_totalvalue['predict_lbl'] - lbe_totalvalue['lable']
lbe_totalvalue.to_excel(r'NOM_10_predict_90.xlsx')
print('#######\n',lbe_totalvalue['type'].value_counts(),'#######\n',lbe_totalvalue['type'].count())






















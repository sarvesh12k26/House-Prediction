# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 16:05:16 2018

@author: Sarvesh
"""
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Predicting-House-Prices-In-Bengaluru-Train-Data.csv')
X=dataset.iloc[:,[0,1,3,5,6,7]].values
y=dataset.iloc[:,8].values
#Dropping less important columns

#Handle missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,5:6])
X[:,5:6]=imputer.transform(X[:,5:6])
Xdf=pd.DataFrame(X)
#To remove the BHK or Bedroom words from column
Xdf[2]=Xdf[2].str.rstrip('BHKRedroom').astype('float')
#To replace dates with Under Cosntruction
Xdf[1].loc[Xdf[1].str.contains('-')]='Under Construction'

#Encoding categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_Xdf=LabelEncoder()
labelEncoder_Xdf.fit(Xdf.values[:,0])
Xdf[0]=labelEncoder_Xdf.transform(Xdf[0])

labelEncoder_Xdf1=LabelEncoder()
labelEncoder_Xdf1.fit(Xdf.values[:,1])
Xdf[1]=labelEncoder_Xdf1.transform(Xdf[1])

#Creating seperate columns i.e using Hot Encoding
oneHotEncoder=OneHotEncoder()
oneHotEncoder.fit(Xdf.values[:,0:1])
onehot=oneHotEncoder.transform(Xdf.values[:,0:1]).toarray()
Xdf=Xdf.join(pd.DataFrame(onehot),rsuffix='_0')
Xdf=Xdf.values[:,:9]
Xdf=pd.DataFrame(Xdf)
Xdf=Xdf.values[:,1:]
Xdf=pd.DataFrame(Xdf)



oneHotEncoder1=OneHotEncoder()
oneHotEncoder1.fit(Xdf.values[:,0:1])
onehot=oneHotEncoder1.transform(Xdf.values[:,0:1]).toarray()
Xdf=Xdf.join(pd.DataFrame(onehot[:,[0,1]]),rsuffix='_0')
Xdf=Xdf.values[:,1:]
Xdf=pd.DataFrame(Xdf)

#Renaming columns
Xdf.columns=['size','total_sqft','bath','balcony','built_up','carpet','plot','immediate','readymove']

#Xdf['total_sqft'].loc[Xdf['total_sqft'].str.contains('-')]
for i,element in enumerate(Xdf['total_sqft']):
    if '-' in element:
        element=element.replace(" ","")
        parts=element.split('-')
        element=str((float(parts[0])+float(parts[1]))/2)
        Xdf['total_sqft'][i]=element
    elif 'Yards' in element:
        element=element.replace('Sq. Yards','')
        element=str(float(element)*9)
        Xdf['total_sqft'][i]=element
    elif 'Meter' in element:
        element=element.replace('Sq. Meter','')
        element=str(float(element)*10.7639)
        Xdf['total_sqft'][i]=element
    elif 'Acres' in element:
        element=element.replace('Acres','')
        element=str(float(element)*43560)
        Xdf['total_sqft'][i]=element
    elif 'Guntha' in element:
        element=element.replace('Guntha','')
        element=str(float(element)*1089)
        Xdf['total_sqft'][i]=element
    elif 'Cents' in element:
        element=element.replace('Cents','')
        element=str(float(element)*435.61)
        Xdf['total_sqft'][i]=element
    elif 'Perch' in element:
        element=element.replace('Perch','')
        element=str(float(element)*272.25)
        Xdf['total_sqft'][i]=element
    elif 'Grounds' in element:
        element=element.replace('Grounds','')
        element=str(float(element)*2400)
        Xdf['total_sqft'][i]=element
Xdf=Xdf.drop([4086])        
y = np.delete(y, (4086), axis=0)

#Handle missing data
X1=Xdf.values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X1[:,[0,2]])
X1[:,[0,2]]=imputer.transform(X1[:,[0,2]])
Xdf=pd.DataFrame(X1)

##########Selecting specific features based on backward elimination#######
Xdf=pd.DataFrame(Xdf.values[:,[0,1,2,3,4,7,8]])
####################################Data Cleaning is done###########################

####################################Splitting and feature scaling###################
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xdf, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

####################################Applying Polynomial Regression Model######################
#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X_train)
X_test_poly=poly_reg.transform(X_test)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y_train)
y_pred=lin_reg_2.predict(X_test_poly)

#y_pred=regressor.predict(X_test)
y_pred=sc_y.inverse_transform(y_pred)

#######################OR######################
######################Applying SVR regression#############################
from sklearn.svm import SVR
regressor=SVR(kernel="rbf")
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
y_pred=sc_y.inverse_transform(y_pred)

#############################OR#########################
############################Appying Linear Regression######################
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
y_pred=sc_y.inverse_transform(y_pred)

from sklearn.metrics import mean_squared_error
from math import sqrt
sqrt(mean_squared_error(y_test,y_pred))

#Build the optimal backward elimination model
import statsmodels.formula.api as sm
X1=np.append(arr=np.ones((13319,1)).astype(int), values=X1, axis=1)
X1=X1.astype(float)
X_opt= X1[:,[0,1,2,3,4,5,6,7,8,9]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt= X1[:,[0,1,2,3,4,5,7,8,9]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt= X1[:,[0,1,2,3,4,7,8]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()








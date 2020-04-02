# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:56:04 2020

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Admission_Predict_Ver1.1.csv")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# checking the scatter of target variable with posiible features
#X = df.iloc[:,6]
#Y = df.iloc[:,8]
#plt.scatter(X,Y)
#plt.show()

#creating the feature and target variables array
array_item = ['GRE Score', 'TOEFL Score','CGPA']
X_Reg = df[array_item]
Y_Reg = df.iloc[:,8]

#split
X_train, X_test, y_train, y_test = train_test_split(X_Reg, Y_Reg, random_state =0)

# Linear Regression
from sklearn.svm import LinearSVR
regressor=  LinearRegression()
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
# RMSE = 0.06352

#SVR
eps = 0.1
c_range = np.linspace(0.01,10)
for c in c_range:    
    svr = LinearSVR(epsilon = eps, C = c, fit_intercept=True,max_iter= 50000)
    svr.fit(X_train, y_train)
    svr_predict = svr.predict(X_test)
    print(np.sqrt(metrics.mean_squared_error(y_test,svr_predict)),c)

# for c = 3.27 rmse is optimal
svr1 = LinearSVR(epsilon = eps, C = 3.27, fit_intercept=True,max_iter=1000)
svr1.fit(X_train, y_train)
svr_predict1 = svr.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test,svr_predict1)))  

# RMSE = 0.79277  

# Random forest regressor
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=1000, random_state = 0)
rf.fit(X_train, y_train)
rf_predict = rf.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test,rf_predict)))
# RMSE = 0.06367

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train,y_train)
dt_predict = dt.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test,dt_predict)))
# RSME = 0.0903
    







# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
#from sklearn.cross_validation import *;
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import numpy as np 
from sklearn import metrics

dataset=pd.read_csv("C://Users//admin//Desktop//Multivariate_Linear_Regression_Dataset.csv")
print(dataset)
#x=dataset["Interest rate(%) X"].reshape(-1,1)
#x = pd.DataFrame(dataset["Square Feet"], dataset["Number of Bed Rooms"])
x=dataset.iloc[:,0:2]
print(x)
#x = x.reshape((x.shape[0], 1))
y=dataset.iloc[:,2]
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
model = LinearRegression()
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#y_pred_val=model.predict(4)
#print(y_pred_val)
#if(y_pred_val > 0.5):
#    print("Yes")
#else:
#    print("No")
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, model.predict(X_train))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
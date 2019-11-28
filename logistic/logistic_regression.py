#https://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.cross_validation import train_test_split
# Glass dataset headers
glass_data_headers = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "glass-type"]
# Loading the Glass dataset in to Pandas dataframe 
glass_data = pd.read_csv("C://Users//admin//Desktop//multi_logistic.csv",names=glass_data_headers)
print "Number of observations :: ", len(glass_data.index)
print "Number of columns :: ", len(glass_data.columns)
print "Headers :: ", glass_data.columns.values

train_x, test_x, train_y, test_y = train_test_split(glass_data[glass_data_headers[:-1]],
    glass_data[glass_data_headers[-1]], train_size=0.7)
# Train multi-class logistic regression model
lr = linear_model.LogisticRegression()
lr.fit(train_x, train_y)
# Train multinomial logistic regression
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)
print "Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, lr.predict(train_x))
print "Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, lr.predict(test_x))
print "Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x))
print "Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x))
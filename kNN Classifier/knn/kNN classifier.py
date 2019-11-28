# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


"""Data Slicing :
Before training the model we have to split the dataset into the training and testing dataset.
To split the dataset for training and testing we are using the sklearn module train_test_split
First of all we have to separate the target variable from the attributes in the dataset.

X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]

Above are the lines from the code which sepearte the dataset. The variable X contains the attributes while the variable Y contains the target variable of the dataset.
Next step is to split the dataset for training and testing purpose.

X_train, X_test, y_train, y_test = train_test_split( 
          X, Y, test_size = 0.3, random_state = 100)

Above line split the dataset for training and testing. As we are spliting the dataset in a ratio of 70:30 between training and testing so we are pass test_size parameter’s value as 0.3.
random_state variable is a pseudo-random number generator state used for random sampling."""
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 
 

# Assigning features and label variables
# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# Label or target varible
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print(weather_encoded)
# converting string labels into numbers
temp_encoded=le.fit_transform(temp)
print(temp_encoded)
label=le.fit_transform(play)
print(label)
#combinig weather and temp into single listof tuples
features=list(zip(weather_encoded,temp_encoded))
print(features)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
# Train the model using the training sets
model.fit(features,label)
#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print(predicted)

#Time to check accuracy
#==============================================================================
# print(kNN_Classifier.score(predictors_test,target_test))
# print(accuracy_score(target_test,prediction,normalize=True))
# print(confusion_matrix(target_test,prediction))
# 
#==============================================================================

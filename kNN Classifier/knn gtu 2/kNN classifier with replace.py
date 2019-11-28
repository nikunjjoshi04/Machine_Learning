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
 

data = pd.read_csv("C://Users//admin//Desktop//knn_dataset.csv")

print(data)
clean_up = {"Outlook" : {"Sunny":0,"Overcast":1,"Rainy":2},
"Temp" : {"Hot":0,"Cool":1,"Mild":2},
"Humidity": {"High":1,"Normal":0},

"Play":{"Yes":1, "No":0}
}
data.replace(clean_up, inplace=True)
print(data)
predictors = data.iloc[:, 1:5]#seggregating the predictor variables
target = data.iloc[:, 5]#Seggregating the target/class variable

# Spliting the dataset into train and test 
predictors_train,predictors_test,target_train,target_test = train_test_split(predictors, target, test_size=0.3, random_state = 123)

kNN_Classifier=KNeighborsClassifier(n_neighbors = 3)
model=kNN_Classifier.fit(predictors_train,target_train)
prediction=kNN_Classifier.predict(predictors_test)
print(kNN_Classifier.predict([1,2,1,False]))
#Time to check accuracy
print(kNN_Classifier.score(predictors_test,target_test))
print(accuracy_score(target_test,prediction,normalize=True))
print(confusion_matrix(target_test,prediction))


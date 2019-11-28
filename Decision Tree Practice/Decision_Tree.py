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
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

data = pd.read_csv("C://Users//admin//Desktop//apndcts.csv")
#print(data)
predictors = data.iloc[:, 0:7]#seggregating the predictor variables
target = data.iloc[:, 7]#Seggregating the target/class variable

# Spliting the dataset into train and test 
predictors_train,predictors_test,target_train,target_test = train_test_split(predictors, target, test_size=0.3, random_state = 123)
dtree_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)
model=dtree_entropy.fit(predictors_train,target_train)
prediction=dtree_entropy.predict(predictors_test)
print(accuracy_score(target_test,prediction,normalize=True))

"""Now that we have a decision tree, we can use the pydotplus package to create a visualization for it.

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())"""

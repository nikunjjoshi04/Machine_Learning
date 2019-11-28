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

import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.tree import export_graphviz



data = pd.read_csv("C:\\Users\\Nikunj\\Downloads\\Documents\\ml_practical\\Practicals\\1\\datset_1_decesiontree.csv")
#print(data)
cleanup_nums = {"Age":     {"Youth": 0, "Middle": 1, "Senior" : 2},
                "Income": {"Low": 0, "Medium": 1, "High" : 2 },
                "Student": {"No": 0, "Yes":1  },
                "Credit Rating": { "Fair": 1, "Excellent" : 2 },
                "Buys-Computer": {"No": 0, "Yes": 1}}
data.replace(cleanup_nums, inplace = True)
print(data)
predictors = data.iloc[:, 1:5]#seggregating the predictor variables
target = data.iloc[:, 5]#Seggregating the target/class variable

# Spliting the dataset into train and test 
predictors_train,predictors_test,target_train,target_test = train_test_split(predictors, target, test_size=0.3, random_state = 123)
dtree_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)
model=dtree_entropy.fit(predictors_train,target_train)
prediction=dtree_entropy.predict(predictors_test)
print("Actual ::" , target_test)
print("Prediction ::" , prediction)
print(accuracy_score(target_test,prediction,normalize=True))
print("Confusion Matrix :",confusion_matrix(target_test, prediction)) 
print ("Accuracy : ", accuracy_score(target_test,prediction)*100) 
print(classification_report(target_test, prediction)) 

# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(model, out_file ='tree.dot', 
               feature_names =['Production Cost'])  

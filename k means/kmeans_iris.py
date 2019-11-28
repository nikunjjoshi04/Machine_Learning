
"""
http://stamfordresearch.com/k-means-clustering-in-python/
Created on Tue Sep 03 13:41:44 2019

@author: admin
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
 
# Only needed if you want to display your plots inline if using Notebook
# change inline to auto if you have Spyder installed

# import some data to play with
iris = datasets.load_iris()
print(iris.data)
print iris.feature_names
print iris.target
print iris.target_names
# Store the inputs as a Pandas Dataframe and set the column names
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']
# Set the size of the plot
#plt.figure(figsize=(14,7))

# Create a colormap
colormap = np.array(['red', 'lime', 'black'])

# K Means Cluster
model = KMeans(n_clusters=3)
model.fit(x)

cluster_labels=model.predict(x)
C= model.cluster_centers_
sil= silhouette_score(x,cluster_labels,metric='euclidean')
print(C)
print(sil)

# # Plot the Models Classifications
plt.subplot(1, 2, 0)
plt.scatter(x.Sepal_Length, x.Sepal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification Sepal')

# # Plot the Models Classifications
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification Petal')
# Performance Metrics
print(sm.accuracy_score(y, cluster_labels))
# Confusion Matrix
print(sm.confusion_matrix(y, cluster_labels))
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 09:21:01 2019

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import datasets


data=pd.read_csv("C://Users//admin//Desktop//column_2C_weka.csv")
print(data.head())
f1=data['pelvic_incidence'].values
f2=data['pelvic_radius'].values
f3=data['sacral_slope'].values

y = pd.DataFrame(data['class'])
y.columns = ['Targets']


X=np.array(list(zip(f1,f2,f3)))
kmeans=KMeans(n_clusters=3,random_state=123)
model=kmeans.fit(X)
cluster_labels=kmeans.predict(X)
C=kmeans.cluster_centers_
sil=silhouette_score(X,cluster_labels,metric='euclidean',sample_size=len(data))
print(C)
print(sil)



#For 2-D plot of the data points along with the Centroids
fig=plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.scatter(C[:,0],C[:,1],marker='*',s=1000)

#For 3-D plot of the data points along with the Centroids
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(X[:,0],X[:,1],X[:,2])
plt.scatter(C[:,0],C[:,1],C[:,2],marker='*',s=1000,c='#050505')

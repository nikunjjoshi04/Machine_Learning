# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn import datasets

iris = datasets.load_iris()
print iris.data
print iris.feature_names
print iris.target
print iris.target_names
# Store the inputs as a Pandas Dataframe and set the column names
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
model.fit(x)
labels = model.labels_
print("Labels:",labels)


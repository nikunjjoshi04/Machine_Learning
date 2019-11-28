import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
iris.feature_names
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df.head(n=5)

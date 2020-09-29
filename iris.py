import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris

df = load_iris()

x= df.data
features=df.feature_names
y= df.target

from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train ,y_test =train_test_split(x ,y , test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train , y_train)

pickle.dump(model, open('iris.pkl', 'wb'))
import math
import pandas as pd
import numpy as np
import warnings
from decisiontree import *
from collections import Counter
from sklearn.model_selection import train_test_split

df = pd.read_csv("Iris.csv")

x = df.drop(["Species","Id"],axis=1)
x

y = df["Species"]
y

X_train, X_test, y_train, y_test = train_test_split(x , y,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)

tree = build_tree(X_train , y_train)

y_pred = predict(tree , X_test)

print(score(y_test,y_pred))

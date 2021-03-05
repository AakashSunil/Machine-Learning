# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
import sys
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Fetch the minst data into X and y
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Normalize the data in the range 0-255 (Pixel Range)
X = X / 255

# Rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:] 
y_train, y_test = y[:60000], y[60000:]

learning_rate_param = float(sys.argv[1])
estimators_param = int(sys.argv[2])
subsamples_param = float(sys.argv[3])
max_depth_param = int(sys.argv[4])
criterion_param = sys.argv[5]
min_samples_split_param = int(sys.argv[6])
min_samples_leaf_param = int(sys.argv[7])
tolerance_param = float(sys.argv[8])

gb_classifier = GradientBoostingClassifier(learning_rate=learning_rate_param,n_estimators=estimators_param,subsample=subsamples_param,max_depth=max_depth_param,criterion=criterion_param,min_samples_split=min_samples_split_param,min_samples_leaf=min_samples_leaf_param,tol=tolerance_param)

gb_classifier.fit(X_train,y_train)

gb_predict = gb_classifier.predict(X_test)
# accuracy
print("Accuracy: \n", accuracy_score(y_true=y_test, y_pred=gb_predict), "\n")
print('Error Rate: ',(1-accuracy_score(y_true=y_test, y_pred=gb_predict))*100)

# cm
print("Confusion Matrix: \n",confusion_matrix(y_true=y_test, y_pred=gb_predict))
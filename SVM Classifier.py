# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
import sys
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.svm import SVC
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

kernel_param = sys.argv[1]
C_param = float(sys.argv[2])
gamma_param = sys.argv[3] if sys.argv[3] == 'scale' or sys.argv[3] == 'auto' else float(sys.argv[3])
degree_param = int(sys.argv[4])
coef0_param = float(sys.argv[5])
cache_size_param = float(sys.argv[6])
tolerance_param = float(sys.argv[7])
max_iter_param = int(sys.argv[8])

svm_classifier = SVC(kernel=kernel_param, C=C_param, gamma=gamma_param,degree=degree_param,coef0=coef0_param,cache_size=cache_size_param,tol=tolerance_param,max_iter=max_iter_param)

svm_classifier.fit(X_train,y_train)

svm_predict = svm_classifier.predict(X_test)
# accuracy
print("Accuracy: \n", accuracy_score(y_true=y_test, y_pred=svm_predict), "\n")
print('Error Rate: ',(1-accuracy_score(y_true=y_test, y_pred=svm_predict))*100)

# cm
print("Confusion Matrix: \n",confusion_matrix(y_true=y_test, y_pred=svm_predict))
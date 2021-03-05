# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
import sys
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.neural_network import MLPClassifier
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

hidden_layers_param = int(sys.argv[1])
activation_param = sys.argv[2]
solver_param = sys.argv[3]
alpha_param = float(sys.argv[4])
batch_size_param = int(sys.argv[5])
tolerance_param = float(sys.argv[6])
learning_rate_param = sys.argv[7]
learning_rate_init_param = float(sys.argv[8])
iterations_param = int(sys.argv[9])

mlp_classifier = MLPClassifier(hidden_layer_sizes=(hidden_layers_param,),activation=activation_param,solver=solver_param,alpha=alpha_param,batch_size=batch_size_param,tol=tolerance_param,learning_rate=learning_rate_param,learning_rate_init=learning_rate_init_param,max_iter=iterations_param)

mlp_classifier.fit(X_train,y_train)

mlp_predict = mlp_classifier.predict(X_test)
# accuracy
print("Accuracy: \n", accuracy_score(y_true=y_test, y_pred=mlp_predict), "\n")
print('Error Rate: ',(1-accuracy_score(y_true=y_test, y_pred=mlp_predict))*100)

# cm
print("Confusion Matrix: \n",confusion_matrix(y_true=y_test, y_pred=mlp_predict))
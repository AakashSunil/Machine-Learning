Machine Learning Assignment - 2b

This assignment was to use the Bag of Words and Bernoulli model to classify text as 'ham' or 'spam' using Logistic Regression Algorithm.
Parameters - learning rate, iterations and the l2 regularizaton value input through the command line by user.

The Folder structure for the code to work accepting command line input is as follows.
Solution Codes
Datasets
|___Dataset 1
|________dataset_name_train
|____________train
|_______________ham
|__________________Text files
|_______________spam
|__________________Text files
|________dataset_name_test
|____________test
|_______________ham
|__________________Text files
|_______________spam
|__________________Text files

Command Line to Execute Logistic Regression is
Python <python_filename> <path_to_the_Parent_Dataset_Folder> <learning_rate> <iterations> <l2_regularization_value>
Ex: python '.\Logistic_Regression.py' '..\Datasets\Dataset 1' 0.5 1000 0.6
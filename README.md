Machine Learning Assignment - 2

This assignment was to use the bag of words and bernoulli model to classify text as 'ham' or 'spam' using Multinomial Naive Bayes and Discrete Naive Bayes Algorithms.

This contains the following
1.	Multinomial Naïve Bayes Solution
2.	Discrete Naïve Bayes Solution

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

Command Line to execute the Naïve Bayes Solutions is

Python <python_filename> <path_to_the_Parent_Dataset_Folder>
Ex: python '.\Multinomial Naive Bayes.py' '..\Datasets\Dataset 1'
Ex: python '.\Bernoulli Naive Bayes.py' '..\Datasets\Dataset 1'

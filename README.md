Machine Learning Assignment - 1

Decision Trees

This is an implementation of Decision Trees learning algorithm based on the ID3 Algorithm in the Tom Mitchell Book.

In this solution, recursion step is done over half the data for each node respectively.
Different Heuristics implemented in this solution - Instead of directly finding the highest information gain, the lowest overall entropy is considered for splitting on best attribute.

Types of tree heuristics implemented

1. With Entropy as the Impurity
2. With Variance as the Impurity
3. With Entropy as the Impurity and Reduced-Error-Pruning
4. With Variance as the Impurity and Reduced-Error-Pruning
5. With Entropy as the Impurity and Depth-Based-Pruning
6. With Variance as the Impurity and Depth-Based-Pruning
7. Random Forest Classifier

Solution Program -> Homework - 1 - Complete.py

Datasets Folder -> all_data

To execute the program - Homework - 1 - Complete.py

The program needs to be where the datasets are stored.
on the terminal type -> python Homework - 1 - Complete.py

The program is Menu Based - Choice for the type of decision tree and choice of the dataset combination.
enter the option number as seen on the menu
similarly, for the dataset choice as well. This choice would select the train,test and validation sets correctly, to avoid mismatch selection due to human error.

The program runs for one type of tree for one set of the dataset only.
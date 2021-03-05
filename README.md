Machine Learning Assignment - 3b

This assignment is to implement three classifiers code using scikit-sklearn.

1. SVM Classifier
2. MLP Classifier
3. Gradient Boosting Classifier

To execute the codes

1. SVM Classifier

python svm_classifier.py <kernel> <C> <Gamma> <Degree> <Coef0> <cache_size> <tolerance> <max_iter>
Ex: python SVM_classifier.py rbf 5 scale 3 0 200 1e-4 500 

2. MLP Classifier

python mlp_classifier.py <hidden layers> <activation function> <solver> <alpha> <batch_size> <tolerance> <learning rate> <learning rate initial> <iterations>
Ex: python MLP_classifier.py 150 tanh adam 0.1 500 1e-3 constant 0.001 1000


3. Gradient Boosting Classifier

python gb_classifier.py <learning rate> <estimators> <subsamples> <max_depth> <criterion> <Minimum Number of Samples for Split> <Minimum Number of Samples for Leaf> <Tolerance>
Ex: python Gradient Boosting Classifier.py 0.5 5000 1 5 mse 5 2 1e-4


The execution time for these codes vary with the hardware setup of the system.
Tuning the parameters should make the code run faster.

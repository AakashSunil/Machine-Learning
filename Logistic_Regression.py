import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import sys
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.metrics import classification_report,recall_score,f1_score,precision_score,accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.model_selection import train_test_split



warnings.filterwarnings('ignore')
def data_retrieval(path):
    fileNameList_Spam_Test=[]
    fileNameList_Ham_Test=[]
    data = []
    data_test = []
    target_test = []
    target = []
    for dirs in os.listdir(path):
        for subdir in os.listdir(os.path.join(path,dirs)):
            if(subdir=='train'):
                train_spam_files = os.listdir(os.path.join(path,dirs,subdir, 'spam'))
                train_ham_files = os.listdir(os.path.join(path,dirs,subdir, 'ham'))
                for spam_file in train_spam_files:
                    with open(os.path.join(path,dirs,subdir, 'spam', spam_file), encoding="latin-1") as f:
                        data.append(f.read())
                        target.append(1)
                    
                for ham_file in train_ham_files:
                    with open(os.path.join(path,dirs,subdir, 'ham', ham_file), encoding="latin-1") as f:
                        data.append(f.read())
                        target.append(0)
            else:
                test_spam_files = os.listdir(os.path.join(path,dirs,subdir, 'spam'))
                test_ham_files = os.listdir(os.path.join(path,dirs,subdir, 'ham'))
                for spam_file in test_spam_files:
                    fileNameList_Spam_Test.append(os.path.join(path,dirs,subdir, 'spam', spam_file))
                    with open(os.path.join(path,dirs,subdir, 'spam', spam_file), encoding="latin-1") as f:
                        data_test.append(f.read())
                        target_test.append(1)
                        
                for ham_file in test_ham_files:
                    fileNameList_Ham_Test.append(os.path.join(path,dirs,subdir, 'spam', spam_file))
                    with open(os.path.join(path,dirs,subdir, 'ham', ham_file), encoding="latin-1") as f:
                        data_test.append(f.read())
                        target_test.append(0)
    
    train_data_list = list(zip(data, target))
    train_data = pd.DataFrame(train_data_list, columns=['email_message_content','label'])

    test_data_list = list(zip(data_test, target_test))
    test_data = pd.DataFrame(test_data_list, columns=['email_message_content','label'])
        
    return train_data, test_data

def cleanDoc(doc):
    tokens = word_tokenize(doc)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    return words

def vocabularyBuilder(data):
    vocabulary = []

    for words in data:
        clean_data = cleanDoc(words)
        for word in clean_data:
            vocabulary.append(word)
    vocabulary = list(set(vocabulary))
    return vocabulary

def bernoulliModel(data, vocabulary):
    word_presence_per_email = {unique_word: [0] * len(data) for unique_word in vocabulary}

    for index, email in enumerate(data):
        for word in cleanDoc(email):
          if word in word_presence_per_email.keys():
            word_presence_per_email[word][index] = 1
          else:
            continue
    return word_presence_per_email

def bagOfWords(data, vocabulary):
    word_counts_per_email = {unique_word: [0] * len(data) for unique_word in vocabulary}

    for index, email in enumerate(data):
        for word in cleanDoc(email):
          if word in word_counts_per_email.keys():
            word_counts_per_email[word][index] += 1
          else:
            continue

    return word_counts_per_email


def sigmoid(score):
  return (1 / (1 + np.exp(-score)))

def predict_probability(features, weights):
  score = np.dot(features, weights)
  return sigmoid(score)

# feature derivative computation with L2 regularization
def l2_feature_derivative(errors, feature, weight, l2_penalty, feature_is_constant):
  derivative = np.dot(np.transpose(errors), feature)
  
  if not feature_is_constant:
    derivative -= 2 * l2_penalty * weight

  return derivative

# log-likelihood computation with L2 regularization
def l2_compute_log_likelihood(features, labels, weights, l2_penalty):
  indicator = (labels==+1)
  scores    = np.dot(features, weights)
  ll        = np.sum((np.transpose(np.array([indicator]))-1)*scores - np.log(1. + np.exp(-scores))) - (l2_penalty * np.sum(weights[1:]**2))
  return ll

# logistic regression with L2 regularization
def l2_logistic_regression(features, labels, lr, epochs, l2_penalty):
  # add bias (intercept) with features matrix
  bias      = np.ones((features.shape[0], 1))
  features  = np.hstack((bias, features))

  # initialize the weight coefficients
  weights = np.zeros((features.shape[1], 1))

  logs = []

  # loop over epochs times
  for epoch in range(epochs):

    # predict probability for each row in the dataset
    predictions = predict_probability(features, weights)

    # calculate the indicator value
    indicators = (labels==+1)

    # calculate the errors
    errors = np.transpose(np.array([indicators])) - predictions

    # loop over each weight coefficient
    for j in range(len(weights)):

      isIntercept = (j==0)

      # calculate the derivative of jth weight cofficient
      derivative = l2_feature_derivative(errors, features[:,j], weights[j], l2_penalty, isIntercept)
      weights[j] += lr * derivative

    # compute the log-likelihood
    ll = l2_compute_log_likelihood(features, labels, weights, l2_penalty)
    logs.append(ll)
    return weights



def lr_with_regularization(X_train, y_train,X_valid,y_valid,X_test,y_test,lr,iterations,l2):
  # hyper-parameters
  learning_Rate = lr
  epochs        = iterations
  l2_Penalty    = l2

  learned_weights = l2_logistic_regression(X_train, y_train, learning_Rate, epochs, l2_Penalty)


  bias_train = np.ones((X_train.shape[0], 1))
  bias_valid = np.ones((X_valid.shape[0], 1))
  bias_test  = np.ones((X_test.shape[0], 1))
  features_train = np.hstack((bias_train, X_train))
  features_test  = np.hstack((bias_test, X_test))
  features_valid = np.hstack((bias_valid, X_valid))

  test_predictions  = (predict_probability(features_test, learned_weights).flatten()>0.4)
  train_predictions = (predict_probability(features_train, learned_weights).flatten()>0.4)
  valid_predictions = (predict_probability(features_valid, learned_weights).flatten()>0.4)

  print("Accuracy of our LR classifier on training data: {}".format(accuracy_score(np.expand_dims(y_train, axis=1), train_predictions)))
  print("Accuracy of our LR classifier on testing data: {}".format(accuracy_score(np.expand_dims(y_test, axis=1), test_predictions)))
  print("Accuracy of our LR classifier on Valid data: {}\n".format(accuracy_score(np.expand_dims(y_valid, axis=1), valid_predictions)))

  # print("Recall Score on training data: {}".format(recall_score(np.expand_dims(y_train, axis=1), train_predictions)))
  # print("Recall Score on testing data: {}\n".format(recall_score(np.expand_dims(y_test, axis=1), test_predictions)))

  # print("Precision on training data: {}".format(precision_score(np.expand_dims(y_train, axis=1), train_predictions)))
  # print("Precision on testing data: {}\n".format(precision_score(np.expand_dims(y_test, axis=1), test_predictions)))

  # print("F1 Score on training data: {}".format(f1_score(np.expand_dims(y_train, axis=1), train_predictions)))
  # print("F1 Score on testing data: {}\n".format(f1_score(np.expand_dims(y_test, axis=1), test_predictions)))

  
  # print("F1 Score on training data: {}".format(classification_report(np.expand_dims(y_train, axis=1), train_predictions)))
  # print("F1 Score on testing data: {}\n".format(classification_report(np.expand_dims(y_test, axis=1), test_predictions)))


path = sys.argv[1]
learning_rate = float(sys.argv[2])
iterations = int(sys.argv[3])
l2_penalty = float(sys.argv[4])

"""
Data Retrieval from the folder structure
"""
train_data, test_data = data_retrieval(path)

"""
Creation of the Vocabulary
"""
vocabulary = vocabularyBuilder(train_data['email_message_content'])

"""
Bernoulli Model Setup - Here it is checked whether the word exists in the document from the vocabulary or not
"""
word_presence_per_email = bernoulliModel(train_data['email_message_content'],vocabulary)
word_presence = pd.DataFrame(word_presence_per_email)
training_data_bernoulli = pd.concat([train_data,word_presence], axis=1)
cols = [col for col in training_data_bernoulli.columns if col not in ['email_message_content', 'label']]
x_train_bernoulli = training_data_bernoulli[cols]
y_train_bernoulli = training_data_bernoulli['label']

x_train_bernoulli_split, x_valid_bernoulli_split,y_train_bernoulli_split,y_valid_bernoulli_split = train_test_split(x_train_bernoulli,y_train_bernoulli,test_size=0.3,random_state=10)

word_presence_per_email_test = bernoulliModel(test_data['email_message_content'],vocabulary)
word_presence_test = pd.DataFrame(word_presence_per_email_test)
testing_data_bernoulli = pd.concat([test_data,word_presence_test], axis=1)

cols = [col for col in testing_data_bernoulli.columns if col not in ['email_message_content', 'label']]
x_test_bernoulli = testing_data_bernoulli[cols]
y_test_bernoulli = testing_data_bernoulli['label']

"""
Bag of Words Setup - Here the frequency of each word is calculated throughout the vocabulary in each mail
"""

word_count_per_email = bagOfWords(train_data['email_message_content'],vocabulary)
word_counts = pd.DataFrame(word_count_per_email)
training_data_bow = pd.concat([train_data,word_counts], axis=1)
cols = [col for col in training_data_bow.columns if col not in ['email_message_content', 'label']]
x_train_bow = training_data_bow[cols]
y_train_bow = training_data_bow['label']


x_train_bow_split, x_valid_bow_split,y_train_bow_split,y_valid_bow_split = train_test_split(x_train_bow,y_train_bow,test_size=0.3,random_state=10)


word_count_per_email_test = bagOfWords(test_data['email_message_content'],vocabulary)
word_counts_test = pd.DataFrame(word_count_per_email_test)
testing_data_bow = pd.concat([test_data,word_counts_test], axis=1)

cols = [col for col in testing_data_bow.columns if col not in ['email_message_content', 'label']]
x_test_bow = testing_data_bow[cols]
y_test_bow = testing_data_bow['label']

print('Results with Bernoulli Model Setup')
lr_with_regularization(x_train_bernoulli_split,y_train_bernoulli_split,x_valid_bernoulli_split,y_valid_bernoulli_split,x_test_bernoulli,y_test_bernoulli,learning_rate,iterations,l2_penalty)

print('Results with Bag of Words Setup')
lr_with_regularization(x_train_bow_split,y_train_bow_split,x_valid_bow_split,y_valid_bow_split,x_test_bow,y_test_bow,learning_rate,iterations,l2_penalty)



from collections import Counter
import numpy as np 
import pandas as pd
import re
import sys
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.metrics import classification_report,recall_score,f1_score,precision_score,accuracy_score

def data_retrieval(path):
    fileNameList_Spam_Train=[]
    fileNameList_Ham_Train=[]
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
            word_presence_per_email[word][index] = 1

    return word_presence_per_email

def bagOfWords(data, vocabulary):
    word_counts_per_email = {unique_word: [0] * len(data) for unique_word in vocabulary}

    for index, email in enumerate(data):
        for word in cleanDoc(email):
            word_counts_per_email[word][index] += 1

    return word_counts_per_email

def calculations(spam_mails,ham_mails,vocabulary,alpha):
    n_words_per_spam_mail = spam_mails['email_message_content'].apply(len)
    n_spam = n_words_per_spam_mail.sum()
    n_words_per_ham_mail = ham_mails['email_message_content'].apply(len)
    n_ham = n_words_per_ham_mail.sum()
    n_vocabulary = len(vocabulary)
    alpha = 1

    parameters_spam = {unique_word:0 for unique_word in vocabulary}
    parameters_ham = {unique_word:0 for unique_word in vocabulary}

    for word in vocabulary:
        n_word_given_spam = spam_mails[word].sum()

        p_word_given_spam = (n_word_given_spam + alpha)/(n_spam + alpha*n_vocabulary)
        parameters_spam[word] = p_word_given_spam

        n_word_given_ham = ham_mails[word].sum()
        p_word_given_ham = (n_word_given_ham + alpha)/(n_ham + alpha * n_vocabulary)
        parameters_ham[word] = p_word_given_ham

    return parameters_spam, parameters_ham


def NBTrainBOW(training_data_BOW,vocabulary):
    spam_mails = training_data_BOW[training_data_BOW['label']==1]
    ham_mails = training_data_BOW[training_data_BOW['label']==0]

    p_spam = len(spam_mails)/len(training_data_BOW)
    p_ham = len(ham_mails)/len(training_data_BOW)

    alpha = 1
    parameters_spam,parameters_ham = calculations(spam_mails,ham_mails,vocabulary,alpha)

    return p_spam,p_ham,parameters_spam,parameters_ham

def NBTrainBernoulli(training_data_Bernoulli,vocabulary):
    spam_mails = training_data_Bernoulli[training_data_Bernoulli['label']==1]
    ham_mails = training_data_Bernoulli[training_data_Bernoulli['label']==0]

    p_spam = len(spam_mails)/len(training_data_Bernoulli)
    p_ham = len(ham_mails)/len(training_data_Bernoulli)

    alpha = 1
    parameters_spam,parameters_ham = calculations(spam_mails,ham_mails,vocabulary,alpha)

    return p_spam,p_ham,parameters_spam,parameters_ham

def classifyEmail(mailContent,p_spam,p_ham,parameters_spam,parameters_ham):
    clean_content = cleanDoc(mailContent)
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in clean_content:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]

        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]

    if p_ham_given_message > p_spam_given_message:
        return 0
    else:
        return 1



path = sys.argv[1]
print(path)

train_data, test_data = data_retrieval(path)


# Creating the vocabulary from the data
vocabulary = vocabularyBuilder(train_data['email_message_content'])


# Bag of Words Representation
word_counts_per_email = bagOfWords(train_data['email_message_content'],vocabulary)
word_counts = pd.DataFrame(word_counts_per_email)
training_data_BOW = pd.concat([train_data,word_counts], axis=1)
print(training_data_BOW.head())

# Bernoulli Model
word_presence_per_email = bernoulliModel(train_data['email_message_content'],vocabulary)
word_presence = pd.DataFrame(word_presence_per_email)
training_data_bernoulli = pd.concat([train_data,word_presence], axis=1)
# print(training_data_bernoulli.head())


# p_spam,p_ham,parameters_spam,parameters_ham = NBTrainBOW(training_data_BOW,vocabulary)
# test_data['predicted'] = test_data['email_message_content'].apply(classifyEmail,args=(p_spam,p_ham,parameters_spam,parameters_ham))
# print(test_data.head())

# correct = 0
# total = test_data.shape[0]

# for row in test_data.iterrows():
#    row = row[1]
#    if row['label'] == row['predicted']:
#       correct += 1

# print('Correct:', correct)
# print('Incorrect:', total - correct)
# print('Accuracy:', correct/total)


p_spam,p_ham,parameters_spam,parameters_ham = NBTrainBernoulli(training_data_bernoulli,vocabulary)
test_data['predicted'] = test_data['email_message_content'].apply(classifyEmail,args=(p_spam,p_ham,parameters_spam,parameters_ham))
print(test_data.head())


print(recall_score(test_data['label'],test_data['predicted']))
print(precision_score(test_data['label'],test_data['predicted']))
print(f1_score(test_data['label'],test_data['predicted']))
print(accuracy_score(test_data['label'],test_data['predicted']))
print(classification_report(test_data['label'],test_data['predicted']))
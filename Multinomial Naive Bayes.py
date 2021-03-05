from collections import Counter
import numpy as np 
import pandas as pd
import sys
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.metrics import classification_report,recall_score,f1_score,precision_score,accuracy_score

class MultinominalNB():
    def __init__(self):
        self.docs = []
        self.classes = []
        self.vocab = []
        self.logprior = dict()
        self.class_vocab = dict()
        self.loglikelihood = dict()
  
    def countCls(self, cls):
        cnt = 0
        for idx, _docs in enumerate(self.docs):
            if (self.classes[idx] == cls):
                cnt += 1

        return cnt

    def buildGlobalVocab(self):
        vocab = []
        for doc in self.docs:
            vocab.extend(self.cleanDoc(doc)) 

        return np.unique(vocab)

    def buildClassVocab(self, _cls):
        curr_word_list = []
        for idx, doc in enumerate(self.docs):
            if self.classes[idx] == _cls:
                curr_word_list.extend(self.cleanDoc(doc))

        if _cls not in self.class_vocab:
            self.class_vocab[_cls]=curr_word_list
        else:
            self.class_vocab[_cls].append(curr_word_list)

    @staticmethod
    def cleanDoc(doc):
        tokens = word_tokenize(doc)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        words = [word for word in stripped if word.isalpha()]

        stop_words = stopwords.words('english')
        words = [w for w in words if not w in stop_words]
        return words

    def fit(self, x, y):
        self.docs = x
        self.classes = y
        num_doc = len(self.docs)
        uniq_cls = np.unique(self.classes)
        self.vocab = self.buildGlobalVocab()
        vocab_cnt = len(self.vocab)
        for _cls in uniq_cls:
            cls_docs_num = self.countCls(_cls)
            self.logprior[_cls] = np.log(cls_docs_num/num_doc)
            self.buildClassVocab(_cls)
            class_vocab_counter = Counter(self.class_vocab[_cls])
            class_vocab_cnt = len(self.class_vocab[_cls])

            for word in self.vocab:
                w_cnt = class_vocab_counter[word]
                self.loglikelihood[word, _cls] = np.log((w_cnt + 1)/(class_vocab_cnt + vocab_cnt))
        
        
    def predict(self,test_docs):
        output = []

        logprior = self.logprior
        vocab = self.vocab
        loglikelihood = self.loglikelihood
        classes = self.classes

        for doc in test_docs:
            uniq_cls = np.unique(classes)
            sum = dict()

            for  _cls in uniq_cls:
                sum[_cls] = logprior[_cls]

                for word in self.cleanDoc(doc):
                    if word in vocab:
                        try:
                            sum[_cls] += loglikelihood[word, _cls]
                        except:
                            print(sum, _cls)

            result = np.argmax(list(sum.values()))
            output.append(uniq_cls[result])

        return output
    


class Implementation():
    def __init__(self):
        self.labels = dict()

    @staticmethod
    def data_retrieval(path):
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
                        with open(os.path.join(path,dirs,subdir, 'spam', spam_file), encoding="latin-1") as f:
                            data_test.append(f.read())
                            target_test.append(1)
                        
                    for ham_file in test_ham_files:
                        with open(os.path.join(path,dirs,subdir, 'ham', ham_file), encoding="latin-1") as f:
                            data_test.append(f.read())
                            target_test.append(0)

        train_data_list = list(zip(data, target))
        train_data = pd.DataFrame(train_data_list, columns=['email_message_content','label'])

        test_data_list = list(zip(data_test, target_test))
        test_data = pd.DataFrame(test_data_list, columns=['email_message_content','label'])
        
        return train_data, test_data

    def main(self,path):
        
        train_data, test_data = self.data_retrieval(path)

        x_train = train_data['email_message_content']
        y_train = train_data['label']

        x_test = test_data['email_message_content']
        y_test = test_data['label']

        nb = MultinominalNB()

        nb.fit(x_train, y_train) 
        predictions_test = nb.predict(x_test)
        print(classification_report(y_test,predictions_test))
        print(recall_score(y_test,predictions_test))
        print(precision_score(y_test,predictions_test))
        print(f1_score(y_test,predictions_test))
        print(accuracy_score(y_test,predictions_test))

if __name__=="__main__":
    path = sys.argv[1]
    Implementation().main(path)
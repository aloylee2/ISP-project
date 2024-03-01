# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:24:20 2021

@author: aloyl
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:13:42 2021

@author: aloyl
"""

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import pickle
import re
from nltk.corpus import stopwords
import nltk
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# rawData = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
# t = []
# for i in range(len(rawData.data)):
#     t.append([rawData.data[i], rawData.target_names[rawData.target[i]]])


nltk.download('stopwords')


data = "C:\\ProjectISP\\Machinelearning\\20newsgroup.json"; #change the path accordingly
data_handler = open(data, "r")
rawdata = pd.read_json(data_handler, orient='records');


data = pd.DataFrame(rawdata, columns=['body','newsgroup'])

#Cleaning Data (Pre-processing) Keep in mind we are targeting body column for this case
print(rawdata.columns)
print("\n")
#We only want this two columns
rawdata = rawdata[['newsgroup','body']]
print(rawdata.columns)
print()

#make lowercase
rawdata['body'] = rawdata['body'].apply(lambda x:x.lower())
print(rawdata.head())
print()

#removing punctuation
rawdata['body'] = rawdata['body'].apply(lambda x:re.sub(r'[^\w\s]',' ',x))
print(rawdata.head())
print()

#stopwords and extra spaces
stop_words = list(stopwords.words('english'))
rawdata['body'] = rawdata['body'].apply(lambda x: " ".join(x for x in x.split(" ") if x not in stop_words))
print(rawdata.head())

#removing numbers
rawdata['body'] = rawdata['body'].apply(lambda x: " ".join(x for x in x.split(" ") if not x.isdigit()))
#removing 1 letters
rawdata['body'] = rawdata['body'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))


train, test = train_test_split(data, test_size = 0.2, stratify = data['newsgroup'], random_state = 42)

train_body = train['body'].to_list()
test_body = test['body'].to_list()

y_train = train['newsgroup'].to_list()
vectorizer = TfidfVectorizer( stop_words='english')
X_train = vectorizer.fit_transform(train_body)

clf = BernoulliNB()
clf.fit(X_train, y_train)

X_test = vectorizer.transform(test_body)
y_test = test['newsgroup'].to_list()

pred = clf.predict(X_test)

# filepathnb='savehere/nb.txt'
# filepathcc='savehere/pui.txt'
# with open (filepathnb,'wb') as pee:
#     pickle.dump(vectorizer,pee)
    
# with open (filepathcc,'wb') as joke:
#     pickle.dump(clf,joke)


# test1, test2 = clf, vectorizer
# with open("savehere/model.txt","wb") as f:
#     pickle.dump(test1, f)
#     pickle.dump(test2, f)
# with open("savehere/model.txt", "rb") as f:
#     testout1 = pickle.load(f)
#     testout2 = pickle.load(f)


    
score = metrics.accuracy_score(y_test, pred)
print("Model accuracy: %0.3f" % score)
print()
print()

print("================ Test Example 1 ================")
test_message = "The Stars were the No. 3 seed in the Western Conference after going 1-2-0 in in the round-robin portion of the Stanley Cup Qualifiers. They defeated the No. 6 seed Calgary Flames in six games in the first round, the No. 2 seed Colorado Avalanche in seven games in the second round and the No. 1 seed Vegas Golden Knights in five games in the conference final to reach the Cup Final for the first time since 2000."
test = vectorizer.transform([test_message])
print(test_message, "====>", clf.predict(test))

print("================ Test Example 2 ================")
test_message = "This machine as CPU running at 9 Mhz with 500MB of RAM"
test = vectorizer.transform([test_message])
print(test_message, "====>", clf.predict(test))
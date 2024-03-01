# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:35:37 2021

@author: aloyl
"""

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
import re
from nltk.corpus import stopwords
import nltk


with open("savehere/model.txt", "rb") as f:
    clf = pickle.load(f)
    vectorizer = pickle.load(f)




print("================ Test Example 1 ================")
test_message = "The Stars were the No. 3 seed in the Western Conference after going 1-2-0 in in the round-robin portion of the Stanley Cup Qualifiers. They defeated the No. 6 seed Calgary Flames in six games in the first round, the No. 2 seed Colorado Avalanche in seven games in the second round and the No. 1 seed Vegas Golden Knights in five games in the conference final to reach the Cup Final for the first time since 2000."
test = vectorizer.transform([test_message])
print(test_message, "====>", clf.predict(test))

print("================ Test Example 2 ================")
test_message = "This machine as CPU running at 9 Mhz with 500MB of RAM"
test = vectorizer.transform([test_message])
print(test_message, "====>", clf.predict(test))

confidence_level = sorted([i * 100 for i in list(clf.predict_proba(test)[0])], reverse=True)
print(test_message, "====>", clf.predict(test), f'({confidence_level[0]})')
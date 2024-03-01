# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:03:39 2021

@author: aloyl
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn import metrics
import pickle
import re
from nltk.corpus import stopwords
import nltk
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt



# rawData = fetch_20categorys(subset='all', remove=('headers', 'footers', 'quotes'))
# t = []
# for i in range(len(rawData.data)):
#     t.append([rawData.data[i], rawData.target_names[rawData.target[i]]])


datas = "D:\\ProjectISP\\MachineLearning\\outputfile.csv" #change the path accordingly
rawdata = pd.read_csv(datas)

# data = pd.concat([data[data['category'] == cat][:200] for cat in set(data['category'])], ignore_index=False)
# data = data.to_csv("C:\\ProjectISP\\MachineLearning\\usethisv2.csv", index=None)
# rawdata = pd.read_csv(data)

datas = pd.DataFrame(rawdata, columns=['html','category'])

datas = datas[['category','html']]

datas['html'] = datas['html'].astype(str)

counts = datas['category'].value_counts()
print(counts)
total = datas.index
print(total)
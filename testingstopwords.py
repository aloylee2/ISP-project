# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:17:21 2021

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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from collections import Counter


# rawData = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
# t = []
# for i in range(len(rawData.data)):
#     t.append([rawData.data[i], rawData.target_names[rawData.target[i]]])




# data = "C:\\ProjectISP\\Machinelearning\\20newsgroup.json"; #change the path accordingly
# data_handler = open(data, "r")
# rawdata = pd.read_json(data_handler, orient='records');


# data = pd.DataFrame(rawdata, columns=['body','newsgroup'])

# #Cleaning Data (Pre-processing) Keep in mind we are targeting body column for this case
# print(rawdata.columns)
# print("\n")
# #We only want this two columns
# rawdata = rawdata[['newsgroup','body']]
# print(rawdata.columns)
# print()

# #make lowercase
# rawdata['body'] = rawdata['body'].apply(lambda x:x.lower())
# print(rawdata.head())
# print()

# #removing punctuation
# rawdata['body'] = rawdata['body'].apply(lambda x:re.sub(r'[^\w\s]',' ',x))
# print(rawdata.head())
# print()

# #removing numbers
# rawdata['body'] = rawdata['body'].apply(lambda x: " ".join(x for x in x.split(" ") if not x.isdigit()))
# #removing 1 letters
# rawdata['body'] = rawdata['body'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))


# filepathbody='body.txt'

# with open (filepathbody,'wb') as pee:
#     pickle.dump(rawdata['body'],pee)

        
# print(rawdata['body'])

# using process_text only
# dict=[]

# with open("alt.atheism.csv", "r") as f:
#     for line in f:
#         if line !="\n":
#             dict.append(line.lower().rstrip())

# cnt = Counter(dict)

# for k, v in cnt.items():
#     print ("the count of " + k + "is:" + str(v))
    
            
            
            
    
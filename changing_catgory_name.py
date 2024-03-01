# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 20:31:21 2021

@author: aloyl
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 01:57:39 2021

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



dataframe = pd.read_csv("D:\\ProjectISP\\MachineLearning\\updated_new.csv")
dataframe = pd.DataFrame(dataframe, columns=['link','category'])

dataframe = dataframe[['category','link']]

dataframe['link'] = dataframe['link'].astype(str)

counts = dataframe['category'].value_counts()
print(counts)
total = dataframe.index
print(total)

dataframe = dataframe.replace(to_replace ="ARTS",value ="ART")
dataframe = dataframe.replace(to_replace ="ARTS & CULTURE",value ="ART")
dataframe = dataframe.replace(to_replace ="CULTURE & ARTS",value ="ART")

dataframe = dataframe.replace(to_replace ="PARENTING",value ="PARENT")

dataframe = dataframe.replace(to_replace ="THE WORLDPOST",value ="WORLDPOST")

dataframe = dataframe.replace(to_replace ="STYLE & BEAUTY",value ="STYLE")

dataframe = dataframe.replace(to_replace ="MONEY",value ="BUSINESS")

dataframe = dataframe.replace(to_replace ="FOOD & DRINK",value ="TASTE")

dataframe = dataframe.replace(to_replace ="GREEN",value ="ENVIRONMENT")

dataframe = dataframe.replace(to_replace ="PARENTS",value ="PARENT")

dataframe = dataframe.replace(to_replace ="COLLEGE",value ="EDUCATION")

dataframe = dataframe.replace(to_replace ="HEALTHY LIVING",value ="WELLNESS")


dataframe.to_csv('D:\\ProjectISP\\MachineLearning\\updated_combined_new.csv',index = False)


datas = "D:\\ProjectISP\\MachineLearning\\updated_combined_new.csv" #change the path accordingly
rawdata = pd.read_csv(datas)

# data = pd.concat([data[data['category'] == cat][:200] for cat in set(data['category'])], ignore_index=False)
# data = data.to_csv("C:\\ProjectISP\\MachineLearning\\usethisv2.csv", index=None)
# rawdata = pd.read_csv(data)

datas = pd.DataFrame(rawdata, columns=['link','category'])

datas = datas[['category','link']]

datas['link'] = datas['link'].astype(str)

counts = datas['category'].value_counts()
print(counts)
total = datas.index
print(total)





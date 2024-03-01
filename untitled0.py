# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:25:44 2022

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


data = pd.read_csv("D:\\ProjectISP\\MachineLearning\\new.csv")
data_cats = set(data["category"].to_list())
data_with_cat = {}

cat_with_num_rows = [
    {"category": "ARTS", "num_rows": 250},
    {"category": "ARTS & CULTURE", "num_rows": 250},
    {"category": "CULTURE & ARTS", "num_rows": 250},
    
    {"category": "PARENTING", "num_rows": 375},
    {"category": "PARENTS", "num_rows": 375},
    
    {"category": "THE WORLDPOST", "num_rows": 375 },
    {"category": "WORLDPOST", "num_rows": 375},
    
    {"category": "STYLE & BEAUTY", "num_rows": 375},
    {"category": "STYLE", "num_rows": 375},
    
    {"category": "MONEY", "num_rows": 375},
    {"category": "BUSINESS", "num_rows": 375},
    
    {"category": "FOOD & DRINK", "num_rows": 375},
    {"category": "TASTE", "num_rows": 375},
    
    {"category": "GREEN", "num_rows": 375},
    {"category": "ENVIRONMENT", "num_rows": 375},
    
    {"category": "COLLEGE", "num_rows": 375},
    {"category": "EDUCATION", "num_rows": 375},
    
    {"category": "HEALTHY LIVING", "num_rows": 375},
    {"category": "WELLNESS", "num_rows": 375},
    

]

data_cats = set(data["category"].to_list())
data_with_cat = {}



for cat in data_cats:
    data_with_cat[cat] = data[data["category"] == cat]
    print(f"{cat}: {len(data_with_cat[cat].index)}")

for cat_with_num_row in cat_with_num_rows:
    try:
        data_with_cat[cat_with_num_row["category"]] = data_with_cat[cat_with_num_row["category"]].head(cat_with_num_row["num_rows"])
        print(f'category "{cat_with_num_row["category"]}" done')
    except Exception:
        print(f'category "{cat_with_num_row["category"]}" does not exist')

for cat in data_cats:
    print(f"{cat}: {len(data_with_cat[cat].index)}")

data = pd.concat([dt for dt in data_with_cat.values()], ignore_index=True)

data.to_csv('D:\\ProjectISP\\MachineLearning\\updated_new.csv',index = False)
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:39:48 2021

@author: aloyl
"""



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



# rawData = fetch_20categorys(subset='all', remove=('headers', 'footers', 'quotes'))
# t = []
# for i in range(len(rawData.data)):
#     t.append([rawData.data[i], rawData.target_names[rawData.target[i]]])


nltk.download('stopwords')

# BBC

data = "C:\\ProjectISP\\Machinelearning\\bbc-text.csv"; #change the path accordingly
rawdata = pd.read_csv ('bbc-text.csv')
print(rawdata)
data = pd.DataFrame(rawdata, columns=['text','category'])
print(len(data.index))
print(set(data['category']))

#Cleaning Data (Pre-processing) Keep in mind we are targeting text column for this case
# print(rawdata.columns)
# print("\n")
# #We only want this two columns
# rawdata = rawdata[['category','text']]
# print(rawdata.columns)
# print()

# #make lowercase
# rawdata['text'] = rawdata['text'].apply(lambda x:x.lower())
# print(rawdata.head())
# print()

# #removing punctuation
# rawdata['text'] = rawdata['text'].apply(lambda x:re.sub(r'[^\w\s]',' ',x))
# print(rawdata.head())
# print()

# #stopwords and extra spaces
# stop_words = list(stopwords.words('english'))
# rawdata['text'] = rawdata['text'].apply(lambda x: " ".join(x for x in x.split(" ") if x not in stop_words))
# print(rawdata.head())

# #removing numbers
# rawdata['text'] = rawdata['text'].apply(lambda x: " ".join(x for x in x.split(" ") if not x.isdigit()))
# #removing 1 letters
# rawdata['text'] = rawdata['text'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
# print(rawdata['text'])
# print()

# # 20Newsgroup

# # data = "C:\\ProjectISP\\Machinelearning\\20newsgroup.json"; #change the path accordingly
# # data_handler = open(data, "r")
# # rawdata = pd.read_json(data_handler, orient='records');


# # data = pd.DataFrame(rawdata, columns=['body','newsgroup'])

# # #Cleaning Data (Pre-processing) Keep in mind we are targeting body column for this case
# # print(rawdata.columns)
# # print("\n")
# # #We only want this two columns
# # rawdata = rawdata[['newsgroup','body']]
# # print(rawdata.columns)
# # print()

# # #make lowercase
# # rawdata['body'] = rawdata['body'].apply(lambda x:x.lower())
# # print(rawdata.head())
# # print()

# # #removing punctuation
# # rawdata['body'] = rawdata['body'].apply(lambda x:re.sub(r'[^\w\s]',' ',x))
# # print(rawdata.head())
# # print()

# # #stopwords and extra spaces
# # stop_words = list(stopwords.words('english'))
# # rawdata['body'] = rawdata['body'].apply(lambda x: " ".join(x for x in x.split(" ") if x not in stop_words))
# # print(rawdata.head())

# # #removing numbers
# # rawdata['body'] = rawdata['body'].apply(lambda x: " ".join(x for x in x.split(" ") if not x.isdigit()))
# # #removing 1 letters
# # rawdata['body'] = rawdata['body'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))


# #training

# train, test = train_test_split(data, test_size = 0.2, stratify = data['category'], random_state = 42)

# train_text = train['text'].to_list()
# test_text = test['text'].to_list()
# print("-------------------------------------------------------------")
# print("-------------------------------------------------------------")
# print(type(train_text))
# print(type(test_text))
# print(train_text)
# print(test_text)

# y_train = train['category'].to_list()
# vectorizer = TfidfVectorizer( stop_words='english')
# X_train = vectorizer.fit_transform(train_text)

# clf = MultinomialNB()
# clf.fit(X_train, y_train)

# X_test = vectorizer.transform(test_text)
# y_test = test['category'].to_list()

# pred = clf.predict(X_test)

# # filepathnb='savehere/nb.txt'
# # filepathcc='savehere/pui.txt'
# # with open (filepathnb,'wb') as pee:
# #     pickle.dump(vectorizer,pee)
    
# # with open (filepathcc,'wb') as joke:
# #     pickle.dump(clf,joke)


# # test1, test2 = clf, vectorizer
# # with open("savehere/model.txt","wb") as f:
# #     pickle.dump(test1, f)
# #     pickle.dump(test2, f)
# # with open("savehere/model.txt", "rb") as f:
# #     testout1 = pickle.load(f)
# #     testout2 = pickle.load(f)


    
# score = metrics.accuracy_score(y_test, pred)
# print("Model accuracy: %0.3f" % score)
# print()
# print()

# print("================ Test Example 1 ================")
# test_message = "The Stars were the No. 3 seed in the Western Conference after going 1-2-0 in in the round-robin portion of the Stanley Cup Qualifiers. They defeated the No. 6 seed Calgary Flames in six games in the first round, the No. 2 seed Colorado Avalanche in seven games in the second round and the No. 1 seed Vegas Golden Knights in five games in the conference final to reach the Cup Final for the first time since 2000."
# test = vectorizer.transform([test_message])
# print(test_message, "====>", clf.predict(test))

# print("================ Test Example 2 ================")
# test_message = "This machine as CPU running at 9 Mhz with 500MB of RAM"
# test = vectorizer.transform([test_message])
# print(test_message, "====>", clf.predict(test))

# print("================ Test Example 3 ================")
# test_message = "One hockey player infected as many as 14 other people at a single indoor ice hockey game last spring, Florida health department officials reported Thursday..That means indoor sports games can turn into superspreader events, the researchers said in the US Centers for Disease Control and Prevention's weekly report.The game was played on June 16 at an ice rink in Tampa and by the following day, a player, considered the index patient, experienced symptoms of Covid-19, including fever, cough, sore throat and a headache. Two days later, he tested positive for the virus, the Florida Department of Health reported.Each team had 11 players, all male, between the ages of 19 and 53, with six on the ice and five on the bench at any given time during the game, the researchers reported. Each team also shared separate locker rooms, typically for 20 minutes before and after the 60-minute game, and no one wore cloth face masks for disease control.During the five days after the game, 15 persons experienced signs and symptoms compatible with coronavirus disease 2019; 13 of the 15 ill persons had positive laboratory test results indicating infection with SARS-CoV-2, the virus that causes COVID-19, researchers wrote. Two of the sick individuals did not get tested.While 62% of the players experienced Covid-19 symptoms, the on-ice referees did not, nor did the one spectator in the stands."
# test = vectorizer.transform([test_message])
# print(test_message, "====>", clf.predict(test))

# print("================ Test Example 2 ================")
# test_message = "A cryptocurrency is a digital or virtual currency that is secured by cryptography, which makes it nearly impossible to counterfeit or double-spend. Many cryptocurrencies are decentralized networks based on blockchain technology—a distributed ledger enforced by a disparate network of computers."
# test = vectorizer.transform([test_message])
# print(test_message, "====>", clf.predict(test))

# print("================ Test Example 2 ================")
# test_message = "Heath is open land with low growing grasses and plants. ... An open, sandy field of low shrubs and scrubby plants like gorse and heather is called a heath. Another word for this kind of uncultivated countryside is moor."
# test = vectorizer.transform([test_message])
# print(test_message, "====>", clf.predict(test))
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



# rawData = fetch_20categorys(subset='all', remove=('headers', 'footers', 'quotes'))
# t = []
# for i in range(len(rawData.data)):
#     t.append([rawData.data[i], rawData.target_names[rawData.target[i]]])


nltk.download('stopwords')

datas = "D:\\ProjectISP\\MachineLearning\\newsgroup_2.csv"#change the path accordingly
rawdata = pd.read_csv(datas)
print(rawdata)

# data = pd.concat([data[data['category'] == cat][:200] for cat in set(data['category'])], ignore_index=False)
# data = data.to_csv("C:\\ProjectISP\\MachineLearning\\usethisv2.csv", index=None)
# rawdata = pd.read_csv(data)

data = pd.DataFrame(rawdata, columns=['link','category'])

data = data[['category','link']]

data['link'] = data['link'].astype(str)
print(data['link'])

counts = data['category'].value_counts()
print(counts)

# # data_cats = set(data["category"].to_list())
# # data = [data[data["category"] == cat] for cat in data_cats]
# # data = [dt.head(750) for dt in data]
# # data = pd.concat(data)

# # counts = data['category'].value_counts()
# # print(counts)

# # x = counts.to_frame().to_csv("oogabooga.csv")
# # plt.hist(x.index, x[category])
# # plt.show()

# # datas = pd.concat([datas[datas['category'] == cat][:599] for cat in set(datas['category'])], ignore_index=False)
# # data = datas.to_csv("D:\\ProjectISP\\MachineLearning\\usethisv2.csv", index=None)


# # data = "D:\\ProjectISP\\MachineLearning\\usethisv2.csv"
# # rawdata = pd.read_csv(data)

# # data = pd.DataFrame(rawdata, columns=['link','category'])

# # data = data[['category','link']]


# # data['link'] = data['link'].astype(str)


# # # remove columns that are less than < number
# # counts = data['category'].value_counts()
# # print(counts)
# # total = data.index
# # print(total)
# # data = data[~data['category'].isin(counts[counts <= 975].index)]


# # data = data.to_csv("C:\\ProjectISP\\MachineLearning\\datasss.csv", index=None)

# #removing numbers
# data['link'] = data['link'].apply(lambda x: " ".join(x for x in x.split(" ") if not x.isdigit()))

# #make lowercase
# data['link'] = data['link'].apply(lambda x:x.lower())

# #removing \n\t 
# data['link'] = data['link'].apply(lambda x:re.sub('\n|\t|\s', ' ', x))

# #removing punctuation
# data['link'] = data['link'].apply(lambda x:re.sub(r'[^\w\s]',' ',x))

# data['link'] = data['link'].apply(lambda x:re.sub('[\w]{2}\d{1}', '',x))
# data['link'] = data['link'].apply(lambda x:re.sub('[\w]{1}\d{1}[\w]{1}', '',x))
# data['link'] = data['link'].apply(lambda x:re.sub('\d{1,}', ' ',x))
# data['link'] = data['link'].apply(lambda x:re.sub('\s[\w]{1}\s', ' ',x))
# data['link'] = data['link'].apply(lambda x:re.sub('\s{1,}', ' ',x))
# data['link'] = data['link'].apply(lambda x:re.sub('^\s{1,}', '',x))


# #stopwords and extra spaces
# stop_words  = list(stopwords.words('english'))
# data['link'] = data['link'].apply(lambda x: " ".join(x for x in x.split(" ") if x not in stop_words))

# print(data.head(10))

# # grouped = data.groupby("category")
# # keys = grouped.groups.keys()
# # for key in keys:
# #     splitdf = grouped.get_group(key)
# #     os.mkdir("C:\\ProjectISP\\MachineLearning\\keywords\\{}\\".format(str(key)))
# #     filelocation = "C:\\ProjectISP\\MachineLearning\\keywords\{}\\".format(str(key))
# #     splitdf.to_csv(filelocation + str(key) + ".csv", index=False)



# train, test = train_test_split(data, test_size = 0.2, random_state = 42)
# train.dropna(axis=0,how='any',inplace=True)
# test.dropna(axis=0,how='any',inplace=True)
# train_text = train['link'].to_list()
# test_text = test['link'].to_list()

# print(train, test)

# # print(train_text)

# y_train = train['category'].to_list()
# vectorizer = TfidfVectorizer( stop_words='english')
# X_train = vectorizer.fit_transform(train_text)

# clf = ComplementNB()
# clf.fit(X_train, y_train)


# X_test = vectorizer.transform(test_text)
# y_test = test['category'].to_list()

# pred = clf.predict(X_test)

# # filepathnb='savehere/nb.txt'
# # filepathcc='savehere/clf.txt'
# # with open (filepathnb,'wb') as nb:
# #     pickle.dump(vectorizer,nb)
    
# # with open (filepathcc,'wb') as clf:
# #     pickle.dump(clf,clf)


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
# test_message = "NEW ORLEANS -- Saints players Alvin Kamara and Cameron Jordan surprised shoppers with a Thanksgiving grocery giveaway on Wednesday night. Kamara and Jordan joined New Orleans restaurateur/entrepreneur Larry Morrow, car dealership owner Matt Bowers and social media personality Supa Cent to pay the bills for everyone in the Save A Lot supermarket, mingle with fans -- and even bag some groceries. The final tally came to about $21,000. We're blessed, so just being able to be a blessing is huge for us, Kamara said. Everybody here is definitely fortunate. This is something so simple, yet so meaningful. Kamara and Morrow had teamed up for Thanksgiving turkey giveaways in past years, though the tradition was interrupted by COVID-19 in 2020. This year they opted to start a new holiday tradition."
# test = vectorizer.transform([test_message])
# print(test_message, "====>", clf.predict(test))
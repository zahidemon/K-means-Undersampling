#!/usr/bin/env python
# coding: utf-8



import numpy as np
import csv
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


file_path= r"C:\Users\win_10\Downloads\Compressed\data_kddcup04\prot_train.csv"
data = csv.reader(open(file_path, encoding='utf-8'))
train = pd.read_csv(file_path)



positive=train[train['0']==1]['0'].count()
negative=train[train['0']==0]['0'].count()
print(positive,negative)


train = train.drop(['279','261532'], axis=1)
train_pos=train[train['0']==1]
train_neg=train[train['0']==0]


kmeans=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=1296, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)


clusters=kmeans.fit(train_neg)


labels=kmeans.labels_
l=[]
for i in range(len(labels)):
    l.append(labels[i])


train_neg['labels']=l
train_neg


train_neg_sampled=train_neg.drop_duplicates('labels')
train_neg_sampled


train_pos=train_pos.append(train_neg_sampled.drop('labels',axis=1))
train_pos


y=train_pos['0']
x=train_pos.drop(['0'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


y_train= train['0']
x_train=train.drop(['0'],axis=1)


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
  
# fitting x samples and y classes 
clf.fit(X_train, y_train) 


y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)







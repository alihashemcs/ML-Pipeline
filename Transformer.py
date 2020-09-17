from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import csv
from array import *
from numpy import *
import numpy as np
#import pandas as pd

w = ['']*4
v = [w]*400

with open('dataCSV1.csv') as dataCSV1:
    csv_reader = csv.reader(dataCSV1, delimiter=',')
    line_count = 0
    count = 0
    for row in csv_reader:
        if(count!=400):
            if line_count == 0:
                line_count += 1
            else:
                v.insert(count,[row[1],row[2],row[4],row[5]])
                line_count += 1
                count += 1
                v.remove(['', '', '', ''])
    print(f'Processed {line_count-1} lines.')

X = np.array(v,dtype=object)
print("before preprocessing: \n",X)
column_trans = ColumnTransformer(
[('enc1', OneHotEncoder(dtype='int'),[1])],
remainder='drop')
column_trans.fit_transform(X)
print("after preprocessing: \n",X)

"""
column_trans = ColumnTransformer(
[('enc1', OneHotEncoder(dtype='int'),[1])],
remainder='passthrough')
"""

"""
column_trans = ColumnTransformer(
[('city_category', OneHotEncoder(dtype='int'),['city'])],
remainder='drop')
"""

"""
column_trans = ColumnTransformer(
[('city_category', OneHotEncoder(dtype='int'),[:,1])],
remainder='drop')
"""
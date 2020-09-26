from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import csv
from array import *
from numpy import *
import numpy as np

clf = RandomForestClassifier(random_state=0)

w = ['']*4
v = [w]*400

with open('dataCSV1.csv') as dataCSV1:
    csv_reader = csv.reader(dataCSV1, delimiter=',')
    line_count = 0
    count = 0
    for row in csv_reader:
        if(count!=400):
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                v.insert(count,[row[1],row[2],row[4],row[5]])
                line_count += 1
                count += 1
                v.remove(['', '', '', ''])
    #print(f'Processed {line_count-1} lines.')

X = np.array(v,dtype=object)
print(X)

column_trans = ColumnTransformer(
			[('source', OneHotEncoder(dtype='int'),[1,2])],
     remainder='passthrough')

Y = column_trans.fit_transform(X)
print(Y)

y = np.zeros(400,dtype=np.int64)

clf.fit(Y,y)

#print(clf.fit(X,y))
#print(clf.predict(X))
#print(clf.predict([[4, 5, 6], [14, 15, 16]]))
print("test")

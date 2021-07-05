import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import csv
from array import *
from numpy import *
import numpy as np

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
print(X)

column_trans = ColumnTransformer(
			[('source', OneHotEncoder(dtype='int'),[1])],
     remainder='passthrough')

Y = column_trans.fit_transform(X)
print(Y)

def main():
	print("Testing the main() test client with command line arguments to test module.")

if __name__ == '__main__' : main()

"""
A = np.zeros((400,4),'U7')

print(A.dtype,A.shape)

for i in range(0,400):
	for j in range(0,4):
		A[i][j] = "hello"
		
print(A)
"""



"""
print("before preprocessing: \n",X)
column_trans = ColumnTransformer(
[('enc1', OneHotEncoder(dtype='int'),[1])],
remainder='drop')
column_trans.fit_transform(X)
print("after preprocessing: \n",X)
"""

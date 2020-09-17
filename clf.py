from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import csv
from array import *
from numpy import *
import numpy as np
#import pandas as pd

def main():
    print("main hello")

def parseCSV():
    w = [''] * 4
    v = [w] * 400
    with open('dataCSV1.csv') as dataCSV1:
        csv_reader = csv.reader(dataCSV1, delimiter=',')
        line_count = 0
        count = 0
        for row in csv_reader:
            if (count != 400):
                if line_count == 0:
                    line_count += 1
                else:
                    v.insert(count, [row[1], row[2], row[4], row[5]])
                    line_count += 1
                    count += 1
                    v.remove(['', '', '', ''])
        print(f'Processed {line_count - 1} lines.')
        return v

def RFC():
    c = parseCSV()
    X = np.array(c, dtype=object)
    #print(X)
    clf = RandomForestClassifier(random_state=0)
    z = []
    for i in range(0, 400):
        z.append(0)  # classes of each sample
    y = np.array(z, dtype=object)
    clf.fit(X, y)
    #print(clf.predict(X))

main()
#parseCSV()
RFC()
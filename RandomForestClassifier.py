from sklearn.ensemble import RandomForestClassifier
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
                #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                #z.append([row[1],row[2],row[4],row[5]])
                v.insert(count,[row[1],row[2],row[4],row[5]])
                line_count += 1
                count += 1
                v.remove(['', '', '', ''])
    #print(f'Processed {line_count-1} lines.')

X = np.array(v,dtype=object)
#print(X)

z = []
for i in range(0,400):
    z.append(0)  # classes of each sample
#print(z)
y = np.array(z,dtype=object)
#print(y)

clf.fit(X,y)
#print(clf.fit(X,y))
#print(clf.predict(X))
#print(clf.predict([[4, 5, 6], [14, 15, 16]]))
print("test")

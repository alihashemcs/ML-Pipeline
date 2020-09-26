import csv
from array import *
from numpy import *
import numpy as np

z = [[]]
w = ['']*4
v = [w]*400
#print(w)
#print(v,"test")

with open('dataCSV1.csv') as dataCSV1:
    csv_reader = csv.reader(dataCSV1, delimiter=',')
    line_count = 0
    count = 0
    for row in csv_reader:
        if(count!=400):
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                #z.append([row[1],row[2],row[4],row[5]])
                v.insert(count,[row[1],row[2],row[4],row[5]])
                line_count += 1
                count += 1
                v.remove(['', '', '', ''])
    print(f'Processed {line_count-1} lines.')

#for i in range(0,400):
#    print(z[i])
#print(z)
#print(v)

X = np.array(v,dtype=object)
print(X.dtype,X.shape,X)

Y = np.array([1,2,3,4])
Z = np.ones(400,dtype=np.int64)
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(Y.dtype,Y.shape)
print(Z.dtype,Z.shape)
print(A.dtype,A.shape)

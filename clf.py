from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import csv
from array import *
from numpy import *
import numpy as np
#import pandas as pd
import parseCSV
import Transformer

def RFC(n):
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

def main():
	x = str(sys.argv[1])
	y = parseCSV.csvToPythonList(x)
	X = parseCSV.pythonListToNumpyArray(y)
	print(X)
	Y = transformColsNumpyArray(X)
	#print(Y)
	print("Testing the main() test client with command line arguments to test module.")
	print(336)
	print(x)

if __name__ == '__main__' : main()

main()
#parseCSV()
RFC()
import sys
import csv
from array import *
from numpy import *
import numpy as np
import pandas as pd

# Import data from csv
# then create numpy array from python list

#Gets name of csv file and puts data into python list
# 400 rows, columns 1,2,4,5 (starting at index 0)
def csvToPythonList(s):
	w = ['']*7
	v = [w]*3919

	with open(s) as dataCSV1:
		csv_reader = csv.reader(dataCSV1, delimiter=',')
		line_count = 0
		count = 0
		for row in csv_reader:
			if(count!=3919):
				if line_count <= 1:
					line_count += 1
				else:
					v.insert(count,[row[0],row[1],row[2],row[3],row[4],row[5],row[6]])
					line_count += 1
					count += 1
					v.remove(['', '', '', '','','',''])
		print(f'Processed {line_count-2} lines.')
	return v

#Python list to numpy array
def pythonListToNumpyArray(l):
	X = np.array(l,dtype=object)
	return X

#numpy array to pandas dataframe
def numpyArrayToPandasDF(l):
	X = pd.DataFrame(l,columns=['No.','Time','SourceIP','Destination','Protocol','Length','Info'])
	X['Time'] = X['Time'].astype('float')
	X['Length'] = X['Length'].astype('int')
	return X

def main():
	x = str(sys.argv[1])
	y = csvToPythonList(x)
	X = pythonListToNumpyArray(y)
	print(X)
	print("Testing the main() test client with command line arguments to test module.")
	print(x)
	Y = numpyArrayToPandasDF(X)
	print(Y)
	print(Y.dtypes)

if __name__ == '__main__' : main()


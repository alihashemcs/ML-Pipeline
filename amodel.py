import sys
import csv
from array import *
from numpy import *
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate

from diffprivlib.models import GaussianNB
import diffprivlib as ibmdp

import sys
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier

######################################################################
##### parseCSV #####
######################################################################
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
		print('Processed' + str(line_count-2) + 'lines.')
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

######################################################################
##### Transformer #####
######################################################################
#Transform columns in numpy array
def transformColsNumpyArray(a):
	#discretize the time feature/column
	#normalize the length feature/column
	#encode sourceIP and protocol features
	t = [('time_disc', KBinsDiscretizer(n_bins=10, encode='onehot'), [0]),
		('source_protocol_enc', OneHotEncoder(dtype='int'), [1,2]),
		('length_norm', Normalizer(), [3])]
	#discretize the time feature/column
	t1 = [('time_disc', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'), [1])]
	#discretize the time feature/column
	#encode all others
	t2 = [('time_disc', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'), [1]),
			('source_protocol_enc', OrdinalEncoder(), [0,2,3,4,5,6])]
	#discretize the time feature/column
	#normalize the length feature/column
	#encode sourceIP and protocol features
	t3 = [('time_disc', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'), [1]),
			('source_protocol_enc', OrdinalEncoder(), [1,2]),
			('length_norm', Normalizer(), [3])]

	column_trans = ColumnTransformer(transformers=t2,
									remainder='passthrough')

	Y = column_trans.fit_transform(a)
	return Y

def main():
    #################### parseCSV ####################
	dataFileName = str(sys.argv[1])
	pythonList = csvToPythonList(dataFileName)
	npArray = pythonListToNumpyArray(pythonList)
	print(npArray)
	print("Testing the main() test client with command line arguments to test module.")
	print(dataFileName)
	pandasDF = numpyArrayToPandasDF(npArray)
	print(pandasDF)
	print(pandasDF.dtypes)

    #################### Transformer ####################
	transformedData = transformColsNumpyArray(pandasDF)
	print(transformedData)
	print("Testing the main() test client with command line arguments to test module.")

if __name__ == '__main__' : main()

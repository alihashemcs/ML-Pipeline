import sys
from diffprivlib.models import GaussianNB
from sklearn.model_selection import cross_validate
import aparseCSV
import aTransformer
import numpy as np
import pandas as pd
import diffprivlib as ibmdp

#Create a GaussianNB model
def createDPGaussianNBModel(X,y):
	dpgnb = GaussianNB()
	dpgnb.fit(X,y)
	return dpgnb

def main():

#Build Model
	#Retieve + Transform input from csv file
	x = str(sys.argv[1])
	X = parseCSV.csvToPythonList(x)
	X = parseCSV.pythonListToNumpyArray(X)
	X = parseCSV.numpyArrayToPandasDF(X)
	X = Transformer.transformColsNumpyArray(X)
	print("Input\n", X)
	#Create data labels array
	y1 = np.zeros(2247,dtype=np.int64)
	y2 = np.ones(1672,dtype=np.int64)
	y = np.concatenate((y1,y2), axis=0)
	print("Labels\n", y)
	#Build model with data and labels
	dpgnb = createDPGaussianNBModel(X,y)

#Test Model using cross validation
	result = cross_validate(dpgnb,X,y)
	test_score = result["test_score"]
	avg = 0
	for i in test_score:
		avg += i
	avg = avg / len(test_score)
	print("CV Result: \n", result)
	print("Average Test Score: \n", avg)

if __name__ == '__main__' : main()

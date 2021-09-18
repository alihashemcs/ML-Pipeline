import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
import parseCSV
import Transformer
import numpy as np

#Create a GaussianNB model
def createGaussianNaiveBayesModel(X,y):
	gnb = GaussianNB()
	gnb.fit(X,y)
	return gnb

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
	gnb = createGaussianNaiveBayesModel(X,y)

#Test Model using cross validation
	result = cross_validate(gnb,X,y)
	print(result)

#Test model with test data (mirai-httpflooding-2-dec.csv)
	z = str(sys.argv[2])
	Z = parseCSV.csvToPythonList(z)
	Z = parseCSV.pythonListToNumpyArray(Z)
	Z = parseCSV.numpyArrayToPandasDF(Z)
	Z = Transformer.transformColsNumpyArray(Z)
	print("Test data\n", Z)
	w = np.ones(3919,dtype=np.int64)
	v=gnb.predict(Z)
	print("Predictions for test data\n", v)
	print("Score\n", gnb.score(Z,w))
	ones=0
	zeros=0
	for i in v:
		if i == 0:
			zeros+=1
		else:
			ones+=1
	print("Ones: \n", ones)
	print("Zeros: \n", zeros)

if __name__ == '__main__' : main()

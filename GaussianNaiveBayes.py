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
	x = str(sys.argv[1])
	X = parseCSV.csvToPythonList(x)
	X = parseCSV.pythonListToNumpyArray(X)
	X = parseCSV.numpyArrayToPandasDF(X)
	
	#print(X)
	X = Transformer.transformColsNumpyArray(X)
	print("Input\n", X)
	
	y1 = np.zeros(2247,dtype=np.int64)
	y2 = np.ones(1672,dtype=np.int64)
	y = np.concatenate((y1,y2), axis=0)
	print("Labels\n", y)
	gnb = createGaussianNaiveBayesModel(X,y)
	#print("Labels\n", kmeans.labels_)
	#print("Cluster centers\n", kmeans.cluster_centers_)
	"""
    #Test model with test data (mirai-httpflooding-2-dec.csv)
	z = str(sys.argv[2])
	Z = parseCSV.csvToPythonList(z)
	Z = parseCSV.pythonListToNumpyArray(Z)
	Z = parseCSV.numpyArrayToPandasDF(Z)
	
	Z = Transformer.transformColsNumpyArray(Z)
	print("Test data\n", Y)

	print("Predictions for test data\n", gnb.predict(Z))
	"""

	result = cross_validate(gnb,X,y)
	
	print("Testing the main() test client with command line arguments to test module.")
	print(result)

if __name__ == '__main__' : main()

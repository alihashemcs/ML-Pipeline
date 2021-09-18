import sys
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate
import parseCSV
import Transformer
import numpy as np

#Create a KMeansClustering model
def createKMeansClustering(X):
	kmeans = KMeans(n_clusters=2, random_state=0)
	kmeans.fit(X)
	return kmeans

def main():
	x = str(sys.argv[1])
	X = parseCSV.csvToPythonList(x)
	X = parseCSV.pythonListToNumpyArray(X)
	X = parseCSV.numpyArrayToPandasDF(X)
	
	#print(X)
	X = Transformer.transformColsNumpyArray(X)
	print("Input\n", X)
	
	kmeans = createKMeansClustering(X)
	print("Labels\n", kmeans.labels_)
	print("Cluster centers\n", kmeans.cluster_centers_)

	print("Predictions\n", kmeans.predict(X))
	
	#result = cross_validate(kmeans,X)
	
	print("Testing the main() test client with command line arguments to test module.")
	#print(result)


if __name__ == '__main__' : main()

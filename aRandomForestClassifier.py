import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import aparseCSV
import aTransformer
import numpy as np

#Create a RandomForestClassifier model
def createRandomForestClassifier(X,y):
	clf = RandomForestClassifier(random_state=0)
	#y = np.zeros(400,dtype=np.int64)
	clf.fit(X,y)
	return clf

#print(clf.fit(X,y))
#print(clf.predict(X))
#print(clf.predict([[4, 5, 6], [14, 15, 16]]))

def main():
	x = str(sys.argv[1])
	X = parseCSV.csvToPythonList(x)
	X = parseCSV.pythonListToNumpyArray(X)
	X = parseCSV.numpyArrayToPandasDF(X)

	#print(X)
	X = Transformer.transformColsNumpyArray(X)
	y = np.zeros(1000,dtype=np.int64)
	print(X)
	print(y)

	clf = createRandomForestClassifier(X,y)
	#print(clf)

	print(clf.predict(X))

	result = cross_validate(clf,X,y)

	print("Testing the main() test client with command line arguments to test module.")
	print(result)


if __name__ == '__main__' : main()

import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import parseCSV
import Transformer

#Create a RandomForestClassifier model
def RandomForestClassifier(Y):
	clf = RandomForestClassifier(random_state=0)
	y = np.zeros(400,dtype=np.int64)
	clf.fit(Y,y)
	return clf

#print(clf.fit(X,y))
#print(clf.predict(X))
#print(clf.predict([[4, 5, 6], [14, 15, 16]]))
print("test")

def main():
	x = str(sys.argv[1])
	y = parseCSV.csvToPythonList(x)
	
	X = parseCSV.pythonListToNumpyArray(y)
	print(X)
	Y = transformColsNumpyArray(X)
	
	y = np.zeros(400,dtype=np.int64)
	
	#print(Y)
	print("Testing the main() test client with command line arguments to test module.")
	print(336)
	print(x)

if __name__ == '__main__' : main()

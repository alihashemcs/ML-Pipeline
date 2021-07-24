import sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
import parseCSV

#Transform columns in numpy array
def transformColsNumpyArray(a):	 
	#discretize the time feature/column
	#normalize the length feature/column
	#encode sourceIP and protocol features 
	column_trans = ColumnTransformer(
				[('time_disc', KBinsDiscretizer(n_bins=10, encode='onehot'), ['time']),
				('source_protocol_enc', OneHotEncoder(dtype='int'), [1,2]),
				('length_norm', Normalizer(), ['length'])],
				remainder='drop')

	Y = column_trans.fit_transform(a)
	return Y

def main():
	x = str(sys.argv[1])
	y = parseCSV.csvToPythonList(x)
	X = parseCSV.pythonListToNumpyArray(y)
	X = parseCSV.numpyArrayToPandasDF(X)
	print(X)
	Y = transformColsNumpyArray(X)
	print(Y)
	print("Testing the main() test client with command line arguments to test module.")
	print(x)

if __name__ == '__main__' : main()

"""
A = np.zeros((400,4),'U7')

print(A.dtype,A.shape)

for i in range(0,400):
	for j in range(0,4):
		A[i][j] = "hello"
		
print(A)
"""



"""
print("before preprocessing: \n",X)
column_trans = ColumnTransformer(
[('enc1', OneHotEncoder(dtype='int'),[1])],
remainder='drop')
column_trans.fit_transform(X)
print("after preprocessing: \n",X)
"""

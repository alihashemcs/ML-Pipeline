import sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
import aparseCSV

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
	x = str(sys.argv[1])
	y = parseCSV.csvToPythonList(x)
	X = parseCSV.pythonListToNumpyArray(y)
	X = parseCSV.numpyArrayToPandasDF(X)
	print(X)
	#print(X.dtypes)
	Y = transformColsNumpyArray(X)
	print(Y)
	print("Testing the main() test client with command line arguments to test module.")
	print(x)

if __name__ == '__main__' : main()

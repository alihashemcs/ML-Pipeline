import sys
import csv
from array import *
from numpy import *
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from diffprivlib.models import GaussianNB
import diffprivlib as ibmdp

# dbenignmiraihttpflooding1dec 2247 benign 1672 malicious

def main():
    ######################################################################
    #################### parseCSV ####################
    ######################################################################
	dataFileName = str(sys.argv[1])
	df = pd.read_csv(dataFileName)
	df = df.astype({'Source': 'string', 'Destination': 'string', 'Protocol': 'string', 'Info': 'string'})
    #################### Create X and y ####################
	X = df
	y1 = np.zeros(2247,dtype=int64)		#first 2247 packets are benign
	y2 = np.ones(1672,dtype=int64)		#second 1672 packets are malicious
	y = pd.DataFrame(data=np.concatenate((y1,y2), axis=0), columns=["Benign/Malicious"], dtype=np.int64)
	print("X and y")
	print(X)
	print(X.dtypes)
	print(y)
	print(y.dtypes)

	#print(X.nunique(axis='rows', dropna=False))

	######################################################################
	#################### Pipeline ####################
	######################################################################
	#normalize length, no., time features
	#encode Source, Destination, Protocol, Info features
	t1_norm = [('normalizer', Normalizer(), [0,1,5]),
				('src_dest_prt_inf_enc', OneHotEncoder(handle_unknown='ignore'), [2,3,4,6])]
	"""normalizer = Normalizer()
	transformedX = normalizer.fit(X)
	transformedX = normalizer.transform(X)
	colTransformer1 = ColumnTransformer(transformers=t1_norm,
										remainder='passthrough')
"""
	#discretize no., time features
	#encode Source, Destination, Protocol, Info features
	t1 = [('no_time_disc', KBinsDiscretizer(n_bins=10, encode='onehot', strategy='uniform'), [0,1])]
	t2 = [('src_dest_prt_inf_enc', OneHotEncoder(handle_unknown='ignore'), [2,3,4,6])]
	t3 = [('time_no_disc', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'), [0,1,5])]
	#setting sparse_threshold=0.3 (or any higher value)
	#produces sparse matrix, set to 0 to produce dense matrix.
	#can also produce dense matrix by setting sparse=False in the OneHotEncoder
	#OneHotEncoder produces 385 columns, +3 more for passthrough
	#^ scipy.sparse matrices
	"""colTransformer = ColumnTransformer(transformers=t3,
									remainder='passthrough',
									sparse_threshold=0.3,
									verbose=True)
	transformedX1 = colTransformer.fit_transform(X)
	#transformedX1 = pd.DataFrame(transformedX1)
	print(colTransformer)
	print(transformedX1)
	#print(colTransformer.get_feature_names_out())
	#print(transformedX1.dtypes)
"""
	#normalize No., Time, Length attributes
	#then discretize No., Time and encode Source, Destination, Protocol, Info
	estimators = [('normalize', ColumnTransformer(transformers=t1_norm, remainder='passthrough', sparse_threshold=0)),
				('disc_enc', ColumnTransformer(transformers=t3, remainder='passthrough', sparse_threshold=0)),
				('clf', GaussianNB())]
	myPipeline = Pipeline(estimators,
						verbose=True)
	[print(key,' : ',value) for key,value in myPipeline.get_params().items()]

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	myPipeline.fit(X_train, y_train)
	print(myPipeline.score(X_test, y_test))


	######################################################################
	#################### Plotting Results ####################
	######################################################################
"""	plt.figure(figsize(12,4))
	df = pd.DataFrame(grid.cv_results_)
	for score in ['mean_test_recall', 'mean_test_precision']:
		plt.plot(
				[_[1] for _ in df['param_class_weight']],
				df[score],
				label=score
		)
	plt.legend()"""

if __name__ == '__main__' : main()

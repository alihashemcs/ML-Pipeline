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
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
#from diffprivlib.models import GaussianNB
#import diffprivlib as ibmdp

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
	#print("X and y")
	#print(X)
	#print(X.dtypes)
	#print(y)
	#print(y.dtypes)
	#print(X.nunique(axis='rows', dropna=False))

	######################################################################
	#################### Pipeline ####################
	######################################################################
	#normalize length, no., time features
	#encode Source, Destination, Protocol, Info features
	t1 = [('normalizer', Normalizer(), [0,1,5]),
		('encoder', OneHotEncoder(handle_unknown='ignore'), [2,3,4,6])]

	#discretize no., time features
	t2 = [('discretizer', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'), [0,1,5])]
	#setting sparse_threshold=0.3 (or any higher value)
	#produces sparse matrix, set to 0 to produce dense matrix.
	#can also produce dense matrix by setting sparse=False in the OneHotEncoder
	#OneHotEncoder produces 385 columns, +3 more for passthrough
	#^ scipy.sparse matrices

	estimators = [('normalize_enc', ColumnTransformer(transformers=t1, remainder='passthrough', sparse_threshold=0)),
				('disc', ColumnTransformer(transformers=t2, remainder='passthrough', sparse_threshold=0)),
				('clf', GaussianNB())]
	myPipeline = Pipeline(steps=estimators, verbose=True)
	#[print(key,' : ',value) for key,value in myPipeline.get_params().items()]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
	#myPipeline.fit(X_train, y_train)
	#print(myPipeline.score(X_test, y_test))

	parameters = [
			{'disc__discretizer__n_bins': [2, 5, 10, 20], 'disc__discretizer__encode': ('onehot', 'onehot-dense', 'ordinal')}
	]
	myGS = GridSearchCV(estimator=myPipeline, param_grid=parameters, n_jobs=-1, error_score=0)
	myGS.fit(X_train, y_train.values.ravel())

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

	#disp = RocCurveDisplay.from_estimator(myGS, X_test, y_test)

if __name__ == '__main__' : main()

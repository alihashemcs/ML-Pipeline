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
	#df.astype({'Source': 'float64'})
	print(df)
	print(df.dtypes)
    #################### Create X and y ####################
	X = df
	y1 = np.zeros(2247,dtype=int8)		#first 2247 packets are benign
	y2 = np.ones(1672,dtype=int8)		#second 1672 packets are malicious
	y = pd.DataFrame(data=np.concatenate((y1,y2), axis=0), columns=["Benign/Malicious"], dtype=np.int8)
	print("X and y")
	print(X)
	print(X.dtypes)
	print(y)
	print(y.dtypes)

	######################################################################
	#################### Grid Search - LogisticRegression ####################
	######################################################################
	LRmodel = LogisticRegression(class_weight={0: 1, 1: 2}, max_iter=1000)
	LRmodel.fit(X,y).predict(X).sum()
	grid = GridSearchCV(
					estimator=LogisticRegression(max_iter=1000),
					param_grid={'class_weight': [{0: 1, 1: v} for v in np.linearspace(1,20,30)]},
					scoring={'precision': make_scorer(precision_score), 'recall_score': make_scorer(recall_score)},
					refit='precision',
					return_train_score=True,
					cv=10,
					n_jobs=-1
			)
	grid.fit(X,y)
	print(pd.DataFrame(grid.cv_results_))

	######################################################################
	#################### Plotting Results ####################
	######################################################################
	plt.figure(figsize(12,4))
	df = pd.DataFrame(grid.cv_results_)
	for score in ['mean_test_recall', 'mean_test_precision']:
		plt.plot(
				[_[1] for _ in df['param_class_weight']],
				df[score],
				label=score
		)
	plt.legend()

if __name__ == '__main__' : main()

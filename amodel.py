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
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from diffprivlib.models import GaussianNB
import diffprivlib as ibmdp

import aparseCSV
import aTransformer

def main():
    ######################################################################
    #################### parseCSV ####################
    ######################################################################
	dataFileName = str(sys.argv[1])
	pythonList = aparseCSV.csvToPythonList(dataFileName)
	npArray = aparseCSV.pythonListToNumpyArray(pythonList)
	#print(npArray)
	#print("Testing the main() test client with command line arguments to test module.")
	print(dataFileName)
	pandasDF = aparseCSV.numpyArrayToPandasDF(npArray)
	#print(pandasDF)
	#print(pandasDF.dtypes)
    #################### Create X and y ####################
	X = pandasDF
	y1 = np.zeros(2247,dtype=int8)		#first 2247 packets are benign
	y2 = np.ones(1672,dtype=int8)		#second 1672 packets are malicious
	y = pd.DataFrame(data=np.concatenate((y1,y2), axis=0), columns=["Benign/Malicious"], dtype=np.int8)
	print("X and y")
	print(X)
	print(X.dtypes)
	print(y)
	print(y.dtypes)

    ######################################################################
    #################### Transformer ####################
    ######################################################################
	#transformedData = aTransformer.transformColsNumpyArray(pandasDF)
	#print(transformedData)
	#print("Testing the main() test client with command line arguments to test module.")

    ######################################################################
    #################### Create Pipeline ####################
    ######################################################################
	myPipeline = Pipeline([
		("scale", StandardScaler()),
		("model", GaussianNB())
	],verbose=True)
	[print(key,' : ',value) for key,value in myPipeline.get_params().items()]
	#myPipeline.fit(X,y)

	######################################################################
    #################### Grid Search Model Selection ####################
    ######################################################################
	myModel = GridSearchCV(estimator=myPipeline,
				param_grid={'model__epsilon' : [1.0,1.1,1.2,1.3,1.4,1.5]},
				cv=3)
	#myModel.fit(X,y)
	#pd.DataFrame(myModel.cv_results_)

if __name__ == '__main__' : main()

import sys
import csv
from array import *
from numpy import *
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate

from diffprivlib.models import GaussianNB
import diffprivlib as ibmdp

import sys
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier

import aparseCSV
import aTransformer

def main():
    ######################################################################
    #################### parseCSV ####################
    ######################################################################
	dataFileName = str(sys.argv[1])
	pythonList = aparseCSV.csvToPythonList(dataFileName)
	npArray = aparseCSV.pythonListToNumpyArray(pythonList)
	print(npArray)
	print("Testing the main() test client with command line arguments to test module.")
	print(dataFileName)
	pandasDF = aparseCSV.numpyArrayToPandasDF(npArray)
	print(pandasDF)
	print(pandasDF.dtypes)

    ######################################################################
    #################### Transformer ####################
    ######################################################################
	transformedData = aTransformer.transformColsNumpyArray(pandasDF)
	print(transformedData)
	print("Testing the main() test client with command line arguments to test module.")

if __name__ == '__main__' : main()

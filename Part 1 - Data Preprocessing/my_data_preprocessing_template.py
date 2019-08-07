# Data Preprocessing Template

# Importing the libraries

#mathematical tools
import numpy as np

#make plots
import matplotlib.pyplot as plt

#import/manage dataets
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

#create matrix of features
X = dataset.iloc[:, :-1].values

#create dependent variable vector
y = dataset.iloc[:, 3].values

#Take care of missing data
#preprocess data, imputer will help us fix missing data

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values='NaN',strategy='mean')
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
print(X[:,0:3])


#Encode categorical data: Country
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#Encode categorical data: Purchased (dependent variable)
labelencoder_Y=LabelEncoder()
y=labelencoder_Y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
# Data Preprocessing Template

#### Importing the libraries ####################

#mathematical tools
import numpy as np
#make plots
import matplotlib.pyplot as plt
#import/manage dataets
import pandas as pd


#### Importing the dataset ####################

dataset = pd.read_csv('Salary_Data.csv')

#create matrix of features
X = dataset.iloc[:, :-1].values

#create dependent variable vector
y = dataset.iloc[:, 1].values


#### Splitting the dataset into the Training set and Test set ####################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


#### Fit Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


#### Predict Test results
y_pred=regressor.predict(X_test)


#### Visualize Training results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('Salary vs. Experience - Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#### Visualize Test results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('Salary vs. Experience - Test Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


"""
#### Feature Scaling ####################
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""
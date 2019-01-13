# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)

#plt.scatter(X,y,color='red')
#plt.plot(X,regressor.predict(X), color='blue')
#plt.title('Truth or bluff LR')
#plt.xlabel("position level")
#plt.xlabel("Salary")


plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or bluff LR')
plt.xlabel("position level")
plt.xlabel("Salary")
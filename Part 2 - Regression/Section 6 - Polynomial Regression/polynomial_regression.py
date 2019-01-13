# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("winequality-white.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,11].values

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((4898,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,6,8,10,11]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor  = LinearRegression();
regressor.fit(X_train,y_train)

y_out = regressor.predict(X_test);

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_opt)
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)


y_pol_out = lin_reg.predict(poly_reg.fit_transform(X_test))


plt.scatter(X,y,color='red')



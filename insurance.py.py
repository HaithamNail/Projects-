# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("insurance.csv")

print(df.info())
x = df.iloc[:, 0:6].values
y = df.iloc[:, -1].values



from sklearn.compose import ColumnTransformer

CT = ColumnTransformer([('first', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
x = CT.fit_transform(x)
from sklearn import linear_model
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(x)
poly.fit(x_poly,y)
o=LinearRegression()
o.fit(x_poly,y)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))






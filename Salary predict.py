# import numpy as np
import pandas as pd
import matplotlip as plot
from sklearn import linear_model

df = pd.read_csv('salary_predict_dataset.csv')

df.fillna(df['test_score'].mean(), inplace=True)
df.fillna(df['interview_score'].mean(), inplace=True)
x = df.iloc[:, 0:3].values
y = df.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("xyz", OneHotEncoder(), [0])], remainder="passthrough")

x = ct.fit_transform(x)
x = x[:, 1:]
# print(x)
# print(np.shape(x))

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
# print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Error: {mean_squared_error(y_test, y_pred)}')
print(y_pred)

import pandas as pd
df=pd.read_csv("employee_data.csv")
df=df.drop(columns='customerID',axis=1)

x=df.iloc[:,:19].values
y=df.iloc[:,19].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([("xyz",OneHotEncoder(),[0])],remainder="passthrough")
x=ct.fit_transform(x)
x=x[:,1:]

ct=ColumnTransformer([("xyz",OneHotEncoder(),[2])],remainder="passthrough")
x=ct.fit_transform(x)
x=x[:,1:]

ct=ColumnTransformer([("xyz",OneHotEncoder(),[3])],remainder="passthrough")
x=ct.fit_transform(x)
x=x[:,1:]

ct=ColumnTransformer([("xyz",OneHotEncoder(),[5])],remainder="passthrough")
x=ct.fit_transform(x)
x=x[:,1:]

ct=ColumnTransformer([("xyz",OneHotEncoder(),[7])],remainder="passthrough")
x=ct.fit_transform(x)
x=x[:,1:]

ct=ColumnTransformer([("xyz",OneHotEncoder(),[9])],remainder="passthrough")
x=ct.fit_transform(x)
x=x[:,1:]
#
ct=ColumnTransformer([("xyz",OneHotEncoder(),[11])])
x=ct.fit_transform(x)
x=x[:,1:]

ct=ColumnTransformer([("xyz",OneHotEncoder(),[12])])
x=ct.fit_transform(x)
x=x[:,1:]

ct=ColumnTransformer([("xyz",OneHotEncoder(),[14])],remainder="passthrough")
x=ct.fit_transform(x)
x=x[:,1:]

ct=ColumnTransformer([("xyz",OneHotEncoder(),[16])],remainder="passthrough")
x=ct.fit_transform(x)
x=x[:,1:]

ct=ColumnTransformer([("xyz",OneHotEncoder(),[18])],remainder="passthrough")
x=ct.fit_transform(x)
x=x[:,1:]


le = LabelEncoder()
y = le.fit_transform
y = pd.DataFrame
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=False)
print(df)
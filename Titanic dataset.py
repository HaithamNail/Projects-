# importing libraries
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

titanic = pd.read_csv('train.csv')
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)
titanic.drop('Name', axis=1, inplace=True)
titanic.drop('PassengerId', axis=1, inplace=True)
titanic.drop('Ticket', axis=1, inplace=True)
titanic.drop('Cabin', axis=1, inplace=True)
lbl_enc = LabelEncoder()
titanic.Sex = lbl_enc.fit_transform(titanic.Sex)
titanic.Embarked = lbl_enc.fit_transform(titanic.Embarked)
# print(titanic.info())

X = titanic.drop('Survived', axis=1)
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training and Prediction
model = LogisticRegression(max_iter=5000)  # Model Building
model.fit(X_train, y_train)  # Model Training
y_pred = model.predict((X_test))  # Model Predicting
print(y_pred)

# Model Evaluation
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)
accuracy_score = accuracy_score(y_test, y_pred)
mean_squared_error = mean_squared_error(y_test, y_pred)
print(f'accuracy_score: {accuracy_score}')
print(f'mean_squared_error: {mean_squared_error}')
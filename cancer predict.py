import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('WDBC.csv')
df['5'].fillna(df['5'].mean(), inplace=True)
print(df.info())
x = df.iloc[:, 1: 10].values
y = df.iloc[:, 10]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

cm = confusion_matrix(y_predict, y_test)
print(cm)
print('DT accuracy_score', accuracy_score(y_test, y_predict))
print('DT mean_squared_error', mean_squared_error(y_test, y_predict))

from sklearn.neighbors import KNeighborsClassifier

classifier_2 = KNeighborsClassifier(n_neighbors=5)
classifier_2.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
# print(y_predict)
print('KNN accuracy_score', accuracy_score(y_test, y_predict))
print('KNN mean_squared_error', mean_squared_error(y_test, y_predict))

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, max_iter=300)
    km.fit(x)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
# plt.show()
km = KMeans(n_clusters=3)
km.fit(x)
print('KMeans accuracy_score', accuracy_score(y_test, y_predict))
print('KMeans mean_squared_error', mean_squared_error(y_test, y_predict))
# plt.scatter(x, y)

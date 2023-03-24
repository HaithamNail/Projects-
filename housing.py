

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('housing2.csv')
df.head()

df.shape

df.isnull().sum()

sns.heatmap(df.corr(), annot=True)

"""<h3>Check the percentage of null in each column</h3>"""

df.isnull().sum()*100/df.shape[0]

"""<h1>Data Preprocessing</h1>

<h2>Remove the NULL Values:</h2>

<h2>housing_median_age Column: </h2>
"""

df["housing_median_age"].unique()

df["housing_median_age"].value_counts()

df["housing_median_age"].describe()

df["housing_median_age"].median()

"""The median is one of the values that already exists in the colmn so I can replace the nulls with the median

<h3>The plot of the column before the preprocessing</h3>
"""

sns.set_palette("RdBu")
sns.displot(data=df, x="housing_median_age", kind="kde")

df['housing_median_age'] = df['housing_median_age'].fillna(df["housing_median_age"].median())

"""<h3>The plot of the column after the preprocessing</h3>"""

sns.displot(data=df, x="housing_median_age", kind="kde")

df.isnull().sum()

"""<h2>households Column:</h2>"""

df["households"].value_counts()

"""the no value have the largest count so I can't replace it with the same value"""

df["households"].isnull().sum()

"""put the value -1 as a temp value for the NAN values"""

df['households'] = df['households'].fillna('-1')

df['households'] = df['households'].replace('no' , '-1')

df["households"].isnull().sum()

"""Convert the datatype of the column to int (to be able to plot the curve)"""

df["households"] = df["households"].astype('int32')

"""Replace the temp value -1 with NAN"""

df['households'] = df['households'].replace(-1 , np.nan)

"""<h3>The plot of the column before the preprocessing</h3>"""

sns.set_palette("Set3")
sns.displot(data=df, x="households", kind="kde")

df["households"].describe()

"""<h3>Fill the null values Randomly</h3>"""

fill_list = df["households"].dropna().unique()
df['households'] = df['households'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

df['households'].isnull().sum()

"""<h3>The plot of the column after the preprocessing</h3>"""

sns.displot(data=df, x="households", kind="kde")

"""The two plots before and after the data preprocessing are almost similar"""

df.isnull().sum()

"""<h2>Population Column</h2>"""

df["population"].isnull().sum()*100/df.shape[0]

"""the percentage of null is 0.2% so I can drop the rows with null values

<h3>The plot of the column before the preprocessing</h3>
"""

sns.set_palette("terrain_r")
sns.displot(data=df, x="population", kind="kde")

df["population"].isnull().sum()

df = df.dropna(subset=['population'])

df["population"].isnull().sum()

"""<h3>The plot of the column after the preprocessing</h3>"""

sns.displot(data=df, x="population", kind="kde")

df.isnull().sum()

"""<h2>median_income Column:</h2>"""

df["median_income"].value_counts()

df["median_income"].describe()

df["median_income"].median()

"""<h3>The plot of the column before the preprocessing</h3>"""

sns.set_palette("deep")
sns.displot(data=df, x="median_income", kind="kde")

"""Fill the nulls with random values to maintain the same distribution"""

fill_list = df["median_income"].dropna().unique()
df['median_income'] = df['median_income'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

df["median_income"].isnull().sum()

"""<h3>The plot of the column after the preprocessing</h3>"""

sns.displot(data=df, x="median_income", kind="kde")

df.isnull().sum()

"""<h2>total_bedrooms Column:</h2>"""

df["total_bedrooms"].value_counts()

"""<h3>The plot of the column before the preprocessing</h3>"""

sns.set_palette("husl")
sns.displot(data=df, x="total_bedrooms", kind="kde")

df["total_bedrooms"].median()

"""Fill the null values with the same distribution using the narmalize in value counts"""

s = df.total_bedrooms.value_counts(normalize=True)

missing = df['total_bedrooms'].isnull()
df.loc[missing,'total_bedrooms'] = np.random.choice(s.index, size=len(df[missing]),p=s.values)

"""<h3>The plot of the column after the preprocessing</h3>"""

sns.displot(data=df, x="total_bedrooms", kind="kde")

df.isnull().sum()

"""<h2>And Finally No NULL Values :) !</h2>

<h2>Remove Outliers:</h2>

<h2>Plot all the Columns to decetect the outliers:</h2>
"""

for col in df.columns:
    if df[col].dtype != 'object':
        bp = sns.boxplot(data = df, x = col)
        plt.show()
        sp = sns.scatterplot(data = df, x = col, y=df["median_house_value"])
        plt.show()

"""<h2>total_rooms Column:</h2>"""

bp = sns.boxplot(data = df, x = "total_rooms")
plt.show()
sp = sns.scatterplot(data = df, x = "total_rooms", y=df["median_house_value"])
plt.show()

Q1_tr = df['total_rooms'].quantile(0.25)
Q3_tr = df['total_rooms'].quantile(0.75)
IQR_tr = Q3_tr - Q1_tr
#changed the default 1.5 to 4 because based on the scatter plot and the box plot not all the values greater than Q3_tr + 1.5*(IQR_tr) are outliers
UB_tr = Q3_tr + 4*(IQR_tr)
df.drop(df[df['total_rooms']>UB_tr].index, axis = 0, inplace =True)

"""<h3>Plot the Column afte Removing the outliers:</h3>"""

bp = sns.boxplot(data = df, x = "total_rooms")
plt.show()
sp = sns.scatterplot(data = df, x = "total_rooms", y=df["median_house_value"])
plt.show()

"""<h2>total_bedrooms Column</h2>"""

bp = sns.boxplot(data = df, x = "total_bedrooms")
plt.show()
sp = sns.scatterplot(data = df, x = "total_bedrooms", y=df["median_house_value"])
plt.show()

Q1_tb = df['total_bedrooms'].quantile(0.25)
Q3_tb = df['total_bedrooms'].quantile(0.75)
IQR_tb = Q3_tb - Q1_tb
#changed the default 1.5 to 3.5 because based on the scatter plot and the box plot not all the values greater than Q3_tr + 1.5*(IQR_tr) are outliers
UB_tb = Q3_tb + 3.5*(IQR_tb)
df.drop(df[df['total_bedrooms']>UB_tb].index, axis = 0, inplace =True)

"""<h3>Plot the Column afte Removing the outliers:</h3>"""

bp = sns.boxplot(data = df, x = "total_bedrooms")
plt.show()
sp = sns.scatterplot(data = df, x = "total_bedrooms", y=df["median_house_value"])
plt.show()

"""<h2>Population Column:</h2>"""

Q1_po = df['population'].quantile(0.25)
Q3_po = df['population'].quantile(0.75)
IQR_po = Q3_po - Q1_po
#changed the default 1.5 to 3 because based on the scatter plot and the box plot not all the values greater than Q3_tr + 1.5*(IQR_tr) are outliers
UB_po = Q3_po + 3*(IQR_po)
df.drop(df[df['population']>UB_po].index, axis = 0, inplace =True)

bp = sns.boxplot(data = df, x = "population")
plt.show()
sp = sns.scatterplot(data = df, x = "population", y=df["median_house_value"])
plt.show()

"""<h3>Plot the Column afte Removing the outliers:</h3>"""

bp = sns.boxplot(data = df, x = "population")
plt.show()
sp = sns.scatterplot(data = df, x = "population", y=df["median_house_value"])
plt.show()

"""<h2>households Column:</h2>"""

bp = sns.boxplot(data = df, x = "households")
plt.show()
sp = sns.scatterplot(data = df, x = "households", y=df["median_house_value"])
plt.show()

Q1_ho = df['households'].quantile(0.25)
Q3_ho = df['households'].quantile(0.75)
IQR_ho = Q3_ho - Q1_ho
#changed the default 1.5 to 5 because based on the scatter plot and the box plot not all the values greater than Q3_tr + 1.5*(IQR_tr) are outliers
UB_ho = Q3_ho + 5*(IQR_ho)
df.drop(df[df['households']>UB_ho].index, axis = 0, inplace =True)

"""<h3>Plot the Column afte Removing the outliers:</h3>"""

bp = sns.boxplot(data = df, x = "households")
plt.show()
sp = sns.scatterplot(data = df, x = "households", y=df["median_house_value"])
plt.show()

"""<h2>median_income Column:</h2>"""

bp = sns.boxplot(data = df, x = "median_income")
plt.show()
sp = sns.scatterplot(data = df, x = "median_income", y=df["median_house_value"])
plt.show()

Q1_me = df['median_income'].quantile(0.25)
Q3_me = df['median_income'].quantile(0.75)
IQR_me = Q3_me - Q1_me
#changed the default 1.5 to 2.5 because based on the scatter plot and the box plot not all the values greater than Q3_tr + 1.5*(IQR_tr) are outliers
UB_me = Q3_me + 2.5*(IQR_me)
df.drop(df[df['median_income']>UB_me].index, axis = 0, inplace =True)

"""<h3>Plot the Column afte Removing the outliers:</h3>"""

bp = sns.boxplot(data = df, x = "median_income")
plt.show()
sp = sns.scatterplot(data = df, x = "median_income", y=df["median_house_value"])
plt.show()

"""<h2>Finally No Outliers :) !</h2>"""

df.info()

"""<h3>Use Label Encoding to deal with categorical data: </h3>"""

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df["ocean_proximity"] = le.fit_transform(df["ocean_proximity"])
df["gender"] = le.fit_transform(df["gender"])

df.info()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True)

"""<h3>Drop Unused Columns and Target column</h3>"""

Y = df["median_house_value"]
X = df.drop(["longitude", "latitude", "gender", "median_house_value"], axis = 1)

"""<h3>Split the data to train and test data</h3>"""

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size= 0.20, random_state=42)

"""<h3>Perform Data Scaling using Robust Scaler</h3>"""

from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
X_Train = rs.fit_transform(X_Train)
X_Test = rs.fit_transform(X_Test)

"""<h3>Linear Regression</h3>"""

from sklearn import linear_model

lr = linear_model.LinearRegression()

lr.fit(X_Train, Y_Train)

lr.score(X_Train, Y_Train)

lr.score(X_Test, Y_Test)

"""the score of the train and test data are almost equal"""

Y_Pred = lr.predict(X_Test)
Y_Test_Pred = pd.DataFrame({"Y_Test": Y_Test, "Y_Pred": Y_Pred})
Y_Test_Pred.head()

Y_Test_Pred = Y_Test_Pred.reset_index()

plt.figure(figsize=(10, 8))
Y_Test_Pred = Y_Test_Pred.drop(["index"], axis=1)
plt.plot(Y_Test_Pred[:50])
plt.legend(["Actual", "Predicted"])

"""<h2>Evaluation:</h2>"""

from sklearn.metrics import r2_score
reg_score = r2_score(Y_Test , Y_Pred)
reg_score

from sklearn.metrics import mean_absolute_error , mean_absolute_percentage_error , mean_squared_error

mse = mean_squared_error(Y_Test , Y_Pred)
print("mse: ", mse)
mape = mean_absolute_percentage_error(Y_Test , Y_Pred)
print("mape: ", mape)
mae = mean_absolute_error(Y_Test , Y_Pred)
print("mae: ", mae)
reg_mse = mean_squared_error(Y_Test , Y_Pred)
reg_rmse  = np.sqrt(reg_mse)
print("reg_mse: ", reg_rmse)

import statsmodels.api as sm

X2 = sm.add_constant(X_Train)
est = sm.OLS(Y_Train , X2)
est2 = est.fit()

print (est2.summary())
# DWM_Practicals
1️⃣ Study Data Mining Tool Weka and Create New ARFF File
Steps:
Install Weka tool on your machine.
Open Weka GUI Chooser.
Go to Explorer.
Click on Open file → Select an existing .arff file or create your own using a text editor.
Save a new file with .arff extension in proper ARFF format.

ARFF Format Example:
arff
Copy
Edit
@RELATION weather

@ATTRIBUTE outlook {sunny, overcast, rainy}
@ATTRIBUTE temperature REAL
@ATTRIBUTE humidity REAL
@ATTRIBUTE windy {TRUE, FALSE}
@ATTRIBUTE play {yes, no}

@DATA
sunny,85,85,FALSE,no
overcast,83,86,FALSE,yes

2️⃣ Perform Treatment of Missing Values in Weka
Steps:
Open your dataset in Weka Explorer.
Go to the Preprocess tab.
Choose the attribute with missing values.
Use Filter → unsupervised → attribute → ReplaceMissingValues.
Apply the filter and save the dataset.

3️⃣ Perform Exploratory Data Analysis (EDA) on Given Dataset
Steps:
Load dataset into Weka or Python (Pandas).

Check:
Number of attributes
Types (Nominal/Numeric)
Missing values
Summary statistics (mean, median, min, max)
Visualize data (histograms, boxplots, scatter plots in Python).

In Python:

python
Copy
Edit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
print(df.describe())
sns.pairplot(df)
plt.show()

4️⃣ Treat Missing Values in Python
Steps:
Load the dataset using Pandas.
Identify missing values:
python
Copy
Edit
df.isnull().sum()

Techniques:
Replace with Mean/Median/Mode
Interpolation
Drop rows/columns

Example:
python
Copy
Edit
df['age'].fillna(df['age'].mean(), inplace=True)

5️⃣ Implement Various Data Flow Transformations (ETL)
Steps:
Load raw data using Pandas.

Apply:
Remove Duplicates
Change data type
Normalize/Scale values
Merge/Join datasets
Create derived columns

Example:
python
Copy
Edit
df.drop_duplicates(inplace=True)
df['salary'] = df['salary']*1.1

6️⃣ Implement OLAP Operations on Multidimensional Data Cube
Steps:
Use Pandas or OLAP library.
Create a Pivot Table:
python
Copy
Edit
pd.pivot_table(df, values='Sales', index=['Region'], columns=['Year'], aggfunc='sum')

Perform:
Roll-up: Group data at a higher level.
Drill-down: Go into deeper details.
Slice: Extract a subset of data cube.
Dice: Select data on multiple dimensions.

7️⃣ Implement Apriori Algorithm
Steps:
Use MLxtend library in Python.
Install:
nginx
Copy
Edit
pip install mlxtend
Load data and apply Apriori:

python
Copy
Edit
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules)

8️⃣ Implement Naïve Bayes Algorithm
Steps:
Load dataset.
Split data into train/test.

Use Scikit-Learn:
python
Copy
Edit
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

9️⃣ Implement K-Nearest Neighbors (KNN) Algorithm
Steps:
Load dataset and preprocess.
Use Scikit-Learn:
python
Copy
Edit
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

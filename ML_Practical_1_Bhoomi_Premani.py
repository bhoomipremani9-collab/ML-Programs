# ML Practical 1
# Extracted from the provided Jupyter Notebook
# Student: Bhoomi Premani
# Section: C2-B1
# Roll No: 2
#
# NOTE: This file contains the code extracted from the uploaded notebook.


# ===== Code Cell 1 =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Code Cell 2 =====
df=pd.read_csv("Titanic-Dataset.csv")
df.head()

# ===== Code Cell 3 =====
df.tail()

# ===== Code Cell 4 =====
print(df.shape)

# ===== Code Cell 5 =====
df.columns

# ===== Code Cell 6 =====
df.info()

# ===== Code Cell 7 =====
df.describe() #numpy used gives only numerical values

# ===== Code Cell 8 =====
df.isnull().sum()

# ===== Code Cell 9 =====
plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(),cbar=False,cmap="viridis")
plt.title("missing values")
plt.show()

# ===== Code Cell 10 =====
df["Age"].fillna(df["Age"].median(),inplace=True)

# ===== Code Cell 11 =====
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)

# ===== Code Cell 12 =====
df.isnull().sum()

# ===== Code Cell 13 =====
df.duplicated().sum()

# ===== Code Cell 14 =====
df.drop("Cabin",axis=1,inplace=True)

# ===== Code Cell 15 =====
sns.boxplot(x=df["Age"])
plt.show()

# ===== Code Cell 16 =====
sns.boxplot(x=df["Fare"])
plt.show()

# ===== Code Cell 17 =====
Q1=df["Age"].quantile(0.25)
Q3=df["Age"].quantile(0.75)

IQR=Q3-Q1

lower=Q1-1.5*IQR
upper=Q3-1.5*IQR

df=df[(df["Age"]>=lower) & (df["Age"]<=upper)]

# ===== Code Cell 18 =====
sns.boxplot(x=df["Age"])
plt.show()

# ===== Code Cell 19 =====
Q1=df["Fare"].quantile(0.25)
Q3=df["Fare"].quantile(0.75)

IQR=Q3-Q1

lower=Q1-1.5*IQR
upper=Q3-1.5*IQR

df=df[(df["Fare"]>=lower) & (df["Fare"]<=upper)]

# ===== Code Cell 20 =====
sns.boxplot(x=df["Fare"])
plt.show()

# ===== Code Cell 21 =====
df["Sex"]=df["Sex"].replace("male",0)
df["Sex"]=df["Sex"].replace("female",0)

# ===== Code Cell 22 =====
df.head()

# ===== Code Cell 23 =====
sns.countplot(x="Sex", data=df)
plt.title("Gender Count")
plt.show()

# ===== Code Cell 24 =====
sns.countplot (x="Sex", hue="Survived", data=df)
plt.title("Gender vs Survival")
plt.show()

# ===== Code Cell 25 =====
sns.scatterplot(x="Age", y = "Fare", data = df)
plt.title("Age vs Fare")
plt.show()
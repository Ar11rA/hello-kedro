import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# just see the sample data
df = pd.read_csv("basics/resources/original.csv")
print('****** Look at data initially start ******')
print(df.head(15))
print(df.tail(15))
print(df.shape)
print(df["target"].unique())
print(df["target"].value_counts())
print(df.isnull().sum(axis=0))
print(df.isnull().sum(axis=1))
print(df.isnull().sum(axis=1).sum())
print(df[df.isnull().sum(axis=1) == 0])
print(df.info())
print('****** Look at data initially end ******')
print("***********")
print("***********")

# summarization
print('****** Summarization start ******')
print(df.describe())
print('****** Summarization end ******')

# histogram
print('****** Histogram start ******')
df.hist(figsize=(15, 10))
plt.show()
print('****** Histogram end ******')

# boxplot
print('****** Boxplot start ******')
df.iloc[:, 0:10].boxplot(figsize=(15, 10))
plt.show()
print('****** Boxplot end ******')

# correlation pairplots
print('****** Correlation pairplots start ******')
sns.pairplot(df.iloc[:, 0:3])
plt.show()
sns.pairplot(df, x_vars=['mean radius', 'mean texture', 'mean area'],
             y_vars=['mean radius', 'mean texture', 'mean area'], hue="target")
plt.show()
print('****** Correlation pairplots end ******')

# correlation matrix
print('****** Correlation matrix start ******')
corr = df.corr()
print(corr)
sns.heatmap(corr, annot=True)
plt.show()
print('****** Correlation matrix end ******')

# stacked histogram
print('****** Stacked histogram start ******')
df.hist(column="mean radius", by="target", figsize=(15, 10))
plt.show()
print('****** Stacked histogram end ******')
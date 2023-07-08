import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/sivabalanb/Data-Analysis-with-Pandas-and-Python/master/nba.csv")
print(df.head(5))
# remove NaN
print(df.shape)
print(df.head(5))
df["Salary"].fillna(0.0, inplace=True)
df.dropna(inplace=True)
print(df.shape)
print(df.head(5))
print(df["Salary"].sum())
print(df[["Number", "Age"]].sum(axis=1))

# sort by salary
df.sort_values("Salary", inplace=True, ascending=False)
print(df.head(5))

# sort by salary and position
df.sort_values(["Salary", "Position"], inplace=True, ascending=[False, True])
print(df.head(5))

# add salary rank to the dataframe
df["SalaryRank"] = df["Salary"].rank(ascending=False).astype("int")
print(df.head(5))

# loc and iloc example
print(df.iloc[0:5])

# set multiple values using loc and condition
df.loc[df["Salary"] > 10000000, ["Team", "Name"]] = "New Team"
print(df.head(5))

# sample method example
print(df.sample(n=5))

# query and where example
print(df.query("Age > 30"))
print(df.where(df["Age"] > 30))

# n smallest and n largest salary
print(df.nsmallest(5, "Salary"))
print(df.nlargest(5, "Salary"))

# apply example
df["Salary"] = df["Salary"].apply(lambda x: x * 2)

# groupby example
df = pd.read_csv("basics/resources/sample.csv")
grouped = df.groupby("department")
print(grouped.groups)
# access a group
print(grouped.get_group("abc"))
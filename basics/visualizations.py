import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])
plt.plot(x, y)
plt.show()

# sns example
plt.figure(figsize=(15, 10))
df = pd.DataFrame({"x": x, "y": y})
sns.lineplot(data=df, x="x", y="y")
plt.show()

df = pd.read_csv("basics/resources/sample.csv")
plt.figure(figsize=(15, 10))
sns.displot(df, x="age", kind="kde")
plt.show()

plt.figure(figsize=(15, 10))
sns.displot(df, x="age", kde=True)
plt.show()

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
sns.displot(grades, kind="kde")
plt.show()
sns.histplot(grades)
plt.show()
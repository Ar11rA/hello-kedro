import pandas as pd

# Series from list
ice_creams_list = ["Vanilla", "Chocolate", "Strawberry", "Rum Raisin", "", None]
ice_creams_series = pd.Series(ice_creams_list)
print(ice_creams_series)
print(ice_creams_series.describe())
print(ice_creams_series.isnull())
print(ice_creams_series.isna())
print(ice_creams_series.dropna()[ice_creams_series != ""])

# Series from dictionary
sushi = {
    "Salmon": "Orange",
    "Tuna": "Red",
    "Uni": "Yellow",
    "Eel": "Orange"
}
sushi_series = pd.Series(sushi)
print(sushi_series)

# Series of numbers
numbers = [1, 3, 26, 15, 9, 15]
numbers_series = pd.Series(numbers)
print(numbers_series.sum())
print(numbers_series.mean())
print(numbers_series.product())
print(numbers_series.median())
print(numbers_series.std())
print(numbers_series.describe())
print(numbers_series.size)
print(numbers_series.value_counts())
print(numbers_series.apply(lambda x: x * 2 if x % 2 == 0 else x * 3))

# Series with index
fruits = ["Apple", "Orange", "Banana", "Grapes", "Pineapple"]
fruits_series = pd.Series(fruits, index=[9, 8, 18, 8, 6])
print(fruits_series)
print(fruits_series.sort_values())
print(fruits_series.sort_index())

# read sample csv and use name column for series
sample_series = pd.read_csv("basics/resources/sample.csv", usecols=["name"]).squeeze("columns")
print(sample_series)
print(sample_series[0])
print(sample_series.get(0))

# print(sample_series[10]) # Error
print(sample_series.get(10))
print(sample_series.get(10, "Name"))

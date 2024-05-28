# Creating a Pandas Series from a list
import pandas as pd

data = [10, 2, 3, 45, 65]
series = pd.Series(data)
print(series)

# Adding a scalar value to each element in the Series
series_add = series + 10
print(series_add)


# Filtering elements in the Series
filtered_series = series[series > 20]
print(filtered_series)

# Calculating the mean of the Series
mean = series.mean()
print(f"The mean value of the series is {mean}")

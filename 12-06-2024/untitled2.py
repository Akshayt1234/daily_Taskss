# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BkoCEna_C7Ng9w4GuW0etITpBUCsH1Wp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data into a DataFrame
df = pd.read_excel('weather_data.xlsx')

# Display the first few rows of the DataFrame
print(df.head(10))

print(df.isnull().sum())

# Handle missing values by filling them with the mean of the column
df.fillna(df.mean(), inplace=True)

# Convert the Date column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Verify the conversion
print(df.dtypes)

# Calculate statistics
mean_temp = np.mean(df['Temperature'])
std_temp = np.std(df['Temperature'])
max_temp = np.max(df['Temperature'])
min_temp = np.min(df['Temperature'])

print(f"Mean Temperature: {mean_temp}")
print(f"Standard Deviation of Temperature: {std_temp}")
print(f"Maximum Temperature: {max_temp}")
print(f"Minimum Temperature: {min_temp}")

# Time series plot of Temperature
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Temperature'], label='Temperature', color='b')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Trend Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Extract month from Date
df['Month'] = df['Date'].dt.month

# Calculate average monthly precipitation
monthly_precipitation = df.groupby('Month')['Precipitation'].mean()

# Bar chart of average monthly precipitation
plt.figure(figsize=(10, 5))
monthly_precipitation.plot(kind='bar', color='c')
plt.xlabel('Month')
plt.ylabel('Average Precipitation (mm)')
plt.title('Average Monthly Precipitation')
plt.grid(True)
plt.show()

# Scatter plot of Wind Speed vs Temperature
plt.figure(figsize=(10, 5))
plt.scatter(df['WindSpeed'], df['Temperature'], alpha=0.6, edgecolors='w', linewidth=0.5)
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Temperature (°C)')
plt.title('Wind Speed vs Temperature')
plt.grid(True)
plt.show()


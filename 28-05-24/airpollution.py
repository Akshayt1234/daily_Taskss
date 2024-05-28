import pandas as pd

# Step 1: Convert the CSV file into a DataFrame
df = pd.read_csv('air-pollution.csv')
print("Original DataFrame:")
print(df)

# Step 2: Create a filter based on the country (India in this case)
filtered_df = df[df['Entity'] == 'India']
print("\nFiltered DataFrame for India:")
print(filtered_df)

# Step 3: Ensure the 'Nitrogen oxide (NOx)' column is numeric
filtered_df['Nitrogen oxide (NOx)'] = pd.to_numeric(filtered_df['Nitrogen oxide (NOx)'], errors='coerce')

# Calculate median, mean, and standard deviation for 'Nitrogen oxide (NOx)' and add them as new columns
filtered_df['NOx_median'] = filtered_df['Nitrogen oxide (NOx)'].median()
filtered_df['NOx_mean'] = filtered_df['Nitrogen oxide (NOx)'].mean()
filtered_df['NOx_std'] = filtered_df['Nitrogen oxide (NOx)'].std()

print("\nDataFrame with median, mean, and std columns added:")
print(filtered_df)

# Step 4: Delete the repeated entries
# Assuming 'Year' uniquely identifies each row for the same country
df_cleaned = filtered_df.drop_duplicates(subset='Year')
print("\nDataFrame after removing duplicate entries:")
print(df_cleaned)

# Step 5: Change the null values into 0
df_cleaned.fillna(0, inplace=True)
print("\nDataFrame after replacing NaN with 0:")
print(df_cleaned)

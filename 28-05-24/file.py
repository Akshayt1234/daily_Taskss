import pandas as pd
df = pd.read_csv(
    'new.csv',
    dtype={'Age': int, 'Salary': float},  # Specify data types for 'age' and 'salary'
    usecols=['Name', 'Age', 'Place']  # Only use the 'name', 'age', and 'place' columns
)
print(df)
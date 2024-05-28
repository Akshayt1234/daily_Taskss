#dataframes is two dimensional which is in a tabular format
import pandas as pd
#create a dataframe with a dictionary
data={
    'Name': ['kingini','tuttu','ikka'],
    'age':[24,23,24],
    'place':['koovode','mavoor','punoor']

}
#convert data into dataframe
df=pd.DataFrame(data)

print(df)
print(df[['Name','place']])
print(df['Name'])
#add a new column into the dataframe
df['stipend']= [1500,500,5000]
#remove a column    
df = df.drop(columns=['age'])
print(df.describe())


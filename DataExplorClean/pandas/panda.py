import pandas as pd

df = pd.read_csv('sample_people_data.csv') #df is a dataframe object
df2 = pd.DataFrame(columns=['ID','Job'],data=[[0,'Bo'],[1,'xx'],[2,'er'],[3,'pp'],[4,'gg'],[6,'tl'],[7,'bl'],[8,'wl'],[9,'ql']])
df3 = df.merge(df2, how='left', on='ID')

print(type(df)) #Dataframe object
print(type(df['Age'])) #Series object
#Each column is a series object 
firstPrint =  False
if( firstPrint):
    print("columns ", df.columns)
    print("Age mean: ",df['age'].mean())
    print("Missing values per column: ",df.isna().sum())
    print("Age sorted: ",df.sort_values('column_name'))
    print("Nth row: ", df.iloc[3])# Gets you the row of the dataframe that is physically the nth row.
    print("Nth row: ", df.loc[3])# gets you the nth row of the dataframe but based on the index of the dataframe
    print(df[['Name', 'Age']])
# Returns the selected columns in adataframe object. You can get as many columns as you want.)
df.drop_duplicates(['ID'])
# Returns a dataframe where the row with thegiven number is removed.
# Same as previous but only takes the given column to check for duplicates values. You can give it more than one column name.

#A dataframe must be though as a whole, not something in which iterate through.
#Think of it in terms of conditions
df['Age in 10 years'] = df['Age'] + 10
df['Gender in binary'] = df['Gender'].apply(lambda x : 1 if x =='M' else( 0 if x =='F' else None) )

df.set_index('Age')
df['Age'].plot()
df.reset_index()

print("Age higher than 40",df['Age'] > 40)
print("Age higher than 40",df[df['Age'] > 40])
# print("Age higher than 40",df[df['Age'] > 40 & df['Gender'] == 'M'])
print("Age higher than 40",df[df['Age'].isin([56,60,49])])


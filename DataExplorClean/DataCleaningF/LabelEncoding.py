from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Color'] = le.fit_transform(df['Color'])

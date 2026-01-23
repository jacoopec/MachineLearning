df = pd.DataFrame({
    'Age': ['25', '30', 'thirty-five'],
    'JoinDate': ['2020-01-01', 'Feb 2020', 'invalid date'],
    'Purchased': ['Yes', 'No', 'Yes']
})

# Convert Age
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Convert JoinDate
df['JoinDate'] = pd.to_datetime(df['JoinDate'], errors='coerce')

# Encode Purchased
df['Purchased'] = df['Purchased'].map({'Yes': 1, 'No': 0})

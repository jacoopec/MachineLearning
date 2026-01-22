import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load data from TXT file
df = pd.read_fwf("beachvolley_data.txt")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from headers

# Encode categorical variables
le_weather     = LabelEncoder()
le_temperature = LabelEncoder()
le_play        = LabelEncoder()

df['Weather_enc']     = le_weather.fit_transform(df['Weather'])
df['Temperature_enc'] = le_temperature.fit_transform(df['Temperature'])
df['Play_enc']        = le_play.fit_transform(df['Play?'])

# Prepare features and target
X = df[['Weather_enc', 'Temperature_enc', 'Number_of_players']]
y = df['Play_enc']

# Train the decision tree classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X, y)

# Function to evaluate new data points
def evaluate_new_data(weather, temperature, num_players):
    w = le_weather.transform([weather])[0]
    t = le_temperature.transform([temperature])[0]
    
    # Create a DataFrame with column names
    input_df = pd.DataFrame([[w, t, num_players]],
                            columns=['Weather_enc', 'Temperature_enc', 'Number_of_players'])
    
    prediction = clf.predict(input_df)[0]
    return le_play.inverse_transform([prediction])[0]


# Example usage
print(evaluate_new_data('Rainy', 'Hot', 6))  # Replace with your own test cases

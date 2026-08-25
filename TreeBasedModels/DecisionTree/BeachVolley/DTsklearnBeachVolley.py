import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# --- Your data ---
data = {
    "Weather":     ["Sunny", "Sunny",  "Sunny", "Windy", "Rainy",  "Rainy",  "Rainy", "Windy",  "Sunny",   "Windy",    "Sunny"],
    "Temperature": ["Cold",  "Normal", "Hot",   "Hot",   "Cold",   "Hot",   "Normal", "Cold",    "Cold",   "Normal",   "Hot"],
    "Play?":       ["No",    "Yes",    "Yes",   "Yes",   "No",     "No",     "No",     "No",     "No",      "No",      "Yes"],
}

df = pd.DataFrame(data)

# --- Encode categorical columns ---
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
print(label_encoders)
# --- Split features and target ---
X = df[["Weather", "Temperature"]]
y = df["Play?"]

# --- Train/Test split (this is where it goes!) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(X_train)

# --- Train model ---
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# --- Example prediction ---
pred = model.predict([[ 
    label_encoders["Weather"].transform(["Sunny"])[0],
    label_encoders["Temperature"].transform(["Cold"])[0]
]])



clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print("Prediction:", label_encoders["Play?"].inverse_transform(pred)[0])

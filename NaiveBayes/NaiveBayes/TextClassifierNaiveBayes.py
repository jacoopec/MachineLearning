from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import pandas as pd

data = {
    "text": [
        "I love this movie",
        "This is a great film",
        "Amazing performance and story",
        "I hate this movie",
        "Terrible acting and bad plot",
        "Worst movie I have ever seen"
    ],
    "label": [
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative"
    ]
}

df = pd.DataFrame(data)



# Vectorize text (bag-of-words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["label"])  # positive=1, negative=0

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Prediction for ", X_test , " ",  y_pred)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

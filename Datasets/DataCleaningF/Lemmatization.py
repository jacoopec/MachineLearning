from nltk.stem import WordNetLemmatizer

filtered  = ["better", "running"]
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered]

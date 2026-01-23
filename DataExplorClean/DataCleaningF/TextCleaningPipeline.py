import re
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already done
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def remove_emojis(text):
    # Remove unicode emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Remove text-based emoticons like :) :P :(
    emoticon_pattern = r'[:;=Xx][\-~]?[)(DPpOo3]'
    text = re.sub(emoticon_pattern, '', text)
    
    return text

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove emojis and emoticons
    text = remove_emojis(text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if you haven't already
import nltk
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess the email text by removing stopwords, non-alphanumeric characters,
    and applying stemming.
    """
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Lowercase text
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\d', '', text)  # Remove digits
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords

    # Perform stemming (reduce words to their root form)
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# NLTK data is downloaded during the build phase (render.yaml buildCommand).
# At runtime we just need to make sure NLTK can find the default data path.
# Fallback: try to download silently if somehow missing.
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


import re
def tokenize(sentence):
    """Split sentence into array of words/tokens using fast regex."""
    return re.findall(r'\w+', sentence.lower())


def stem(word):
    """Return the stemmed (root) form of a word."""
    stemmer = nltk.PorterStemmer()
    return stemmer.stem(word.lower())


memo = {}
def lemmatize(word):
    """Return the lemmatized (base) form of a word."""
    w = word.lower()
    if w not in memo:
        memo[w] = lemmatizer.lemmatize(w)
    return memo[w]


def bag_of_words(tokenized_sentence, words):
    """
    Return a bag-of-words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    """
    sentence_words = set([lemmatize(w.lower()) for w in tokenized_sentence if w.isalnum()])
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

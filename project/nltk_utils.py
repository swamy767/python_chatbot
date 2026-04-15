import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Auto-download required NLTK resources for cloud deployment
nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)
nltk.download('punkt_tab', download_dir=NLTK_DATA_DIR, quiet=True)
nltk.download('wordnet', download_dir=NLTK_DATA_DIR, quiet=True)
nltk.download('omw-1.4', download_dir=NLTK_DATA_DIR, quiet=True)

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    # Using the PorterStemmer for stemming
    stemmer = nltk.PorterStemmer()
    return stemmer.stem(word.lower())

def lemmatize(word):
    """
    Lemmatization = find the base form of the word
    examples:
    words = ["am", "is", "are", "was", "were"]
    words = [lemmatize(w) for w in words]
    -> ["be", "be", "be", "be", "be"]
    """
    return lemmatizer.lemmatize(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [lemmatize(word.lower()) for word in tokenized_sentence if word.isalnum()]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

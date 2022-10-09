import nltk
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def tokenize(s):
    return nltk.word_tokenize(s)

def stem(word):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return stemmer.stem(word.lower())

def bag_of_words(t_s, w):
    t_s = [stem(w) for w in t_s]
    bag = np.zeros(len(w), dtype=np.float32)

    for i, w in enumerate(w):
        if w in t_s:
            bag[i] = 1.0
    return bag
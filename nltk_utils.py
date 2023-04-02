from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt')

ps = PorterStemmer()
cv = CountVectorizer()


# split sentence into array of words/tokens
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Finding the root form of the word
def stem(word):
    return ps.stem(word.lower())


# Creating the Bag of Words model
def bag_of_words(corpus):
    return cv.fit_transform(corpus).toarray()


# Applying the Bag of Words model
def apply_bag_of_words(corpus):
    return cv.transform(corpus).toarray()

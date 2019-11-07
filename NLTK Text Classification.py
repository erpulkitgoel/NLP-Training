import nltk
import random

from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB


import pickle

documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


random.shuffle(documents)
#print(documents[1])


all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))


print(all_words["stupid"])

 
word_features = list(all_words.keys())[:3000]
#word_features = list(all_words.keys())[:100]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)   # this is a boolean question True/false

    return features


#print((find_features(movie_reviews.words('neg/cv001_19502.txt'))))


featuresets = [(find_features(rev),category) for (rev,category) in documents]


training_set = featuresets[:1900]
testing_set = featuresets[1900:]
print("testing_set",testing_set)
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Baiyes accuracy test: " , (nltk.classify.accuracy(classifier,testing_set))*100)


classifier.show_most_informative_features(15)





















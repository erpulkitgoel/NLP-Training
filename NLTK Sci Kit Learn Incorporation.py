import nltk
import random

from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

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


#######################print(all_words["stupid"])

 
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
#######print("testing_set",testing_set)
classifier = nltk.NaiveBayesClassifier.train(training_set)



print("Original Naive Baiyes accuracy test: " , (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)



MultiNomialNB_Classifier = SklearnClassifier(MultinomialNB())
MultiNomialNB_Classifier.train(training_set)
print("Multi Nomial Naive Baiyes accuracy test: " , (nltk.classify.accuracy(MultiNomialNB_Classifier,testing_set))*100)

##GausssianNB_Classifier = SklearnClassifier(GaussianNB())
##GausssianNB_Classifier.train(training_set)
##print("Gaussian Naive Baiyes accuracy test: " , (nltk.classify.accuracy(GausssianNB_Classifier,testing_set))*100)

BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_Classifier.train(training_set)
print("Bernoulli Naive Baiyes accuracy test: " , (nltk.classify.accuracy(BernoulliNB_Classifier,testing_set))*100)



#LogisticRegression, SGDClassifier
#SVC, LinearSVC, NuSVC

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_set)
print("LogisticRegression accuracy test: " , (nltk.classify.accuracy(LogisticRegression_Classifier,testing_set))*100)


SGDClassifier_Classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_Classifier.train(training_set)
print("SGDClassifier accuracy test: " , (nltk.classify.accuracy(SGDClassifier_Classifier,testing_set))*100)


SVC_Classifier = SklearnClassifier(SVC())
SVC_Classifier.train(training_set)
print("SVC accuracy test: " , (nltk.classify.accuracy(SVC_Classifier,testing_set))*100)


LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print("LinearSVCs accuracy test: " , (nltk.classify.accuracy(LinearSVC_Classifier,testing_set))*100)


NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print("NuSVC accuracy test: " , (nltk.classify.accuracy(NuSVC_Classifier,testing_set))*100)













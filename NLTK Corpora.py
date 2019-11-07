import nltk

print(nltk.__file__)


from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)

print(tok[1:10])

print(tok[0:10])

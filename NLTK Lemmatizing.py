from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better",pos="a"))
print(lemmatizer.lemmatize("best",pos="a"))
print(lemmatizer.lemmatize("good",pos="a"))
print(lemmatizer.lemmatize("run",pos="n"))
print(lemmatizer.lemmatize("running",pos="n"))
print(lemmatizer.lemmatize("cacti",pos="n"))

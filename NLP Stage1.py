from nltk.tokenize import sent_tokenize, word_tokenize
# tokenizing :Word Tokenizer ....sentence Tokennizer
# lexicon and corporas
# Corporas-- Body of text. ex: medical journal, presidential speeches, English Language
# lexicon -- words and their meanings

# investor Speak and regular english Speak
# investor speak 'bull' : someone who is positive about market
# english speka 'bull' : scary animal you dont want to fight with

example_text= " Hello Mr Pulkit, how are you doing today? The weather is fine and python is owesome. The sky is also clear"

print(sent_tokenize(example_text))
print(word_tokenize(example_text))


for i in word_tokenize(example_text):
    print(i)


# """"""""""""""""Stop Words """"""""""""""""

from nltk.corpus import stopwords

example_sentence = " This is a sentence showing Stop words filtration."

stop_words = set(stopwords.words("english"))
print(stop_words)

words = word_tokenize(example_sentence)

filtered_words = []

for w in words:
    if w not in stop_words:
        filtered_words.append(w)

print("filtered_words" , filtered_words)




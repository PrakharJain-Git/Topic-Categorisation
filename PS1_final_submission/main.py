import nltk
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
nltk.download('wordnet')


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')



import gensim
import string
from gensim import corpora



nltk.download('punkt')


text1 = "Text classification machine learning is one of the most commonly used NLP tasks. In this article, we saw a machine learning simple example of machine learning how text classification machine learning can be performed machine learning in Python. We performed the sentimental analysis of movie reviews. We loaded machine learning machine learning machine learning our trained model and stored it in the model variable. Let's predict the sentiment for the test set using o machine learning machine"
tokens = nltk.word_tokenize(text1)
lowercase_tokens = [t.lower() for t in tokens]
#print(lowercase_tokens)


bagofwords_1 = Counter(lowercase_tokens)
print("Top 10 most common words" , bagofwords_1.most_common(10))
print()


alphabets = [t for t in lowercase_tokens if t.isalpha()]

words = stopwords.words("english")
stopwords_removed = [t for t in alphabets if t not in words]

print("Excluding the stopwords if any", stopwords_removed)
print()


lemmatizer = WordNetLemmatizer()

lem_tokens = [lemmatizer.lemmatize(t) for t in stopwords_removed]

bag_words = Counter(lem_tokens)


stopwords = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(document):
    stopwordremoval = " ".join([i for i in document.lower().split() if i not in stopwords])
    punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punctuationremoval.split())
    return normalized




final_doc = [clean(document).split() for document in bag_words]


# print("Before text-cleaning:", bag_words)

print("After text-cleaning:",final_doc)
print()


dictionary2 = corpora.Dictionary(final_doc)
#print(dictionary2)
DT_matrix2 = [dictionary2.doc2bow(doc) for doc in final_doc]
#print (DT_matrix)

Lda_object_2 = gensim.models.ldamodel.LdaModel



from gensim import models
num_topics2 = 20
lda_model2 = models.LdaModel(DT_matrix2, num_topics=num_topics2, \
                                  id2word=dictionary2, \
                                  passes=5, alpha=[0.01]*num_topics2, \
                                  eta=[0.01]*len(dictionary2.keys()))


bow_vector = dictionary2.doc2bow(lem_tokens)
for index, score in sorted(lda_model2[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model2.print_topic(index, 5)))
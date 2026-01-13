documents=["I love machine learning",
           "I love deep learning",
           "machine learning is great",]

#covert to lowescase and split
tokenized_docs = [doc.lower().split() for doc in documents]

# Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
bow_count= count_vectorizer.fit_transform(documents)

print("Bag of Words Model:(count)")
print(bow_count.toarray())
print()

# Bag of word (Normalized count)

bow_normalized = bow_count.toarray().astype('float')

for i in range(len(bow_normalized)):
    total_words = bow_normalized[i].sum()
    if total_words != 0:
        bow_normalized[i] = bow_normalized[i] / total_words

print("Bag of Words Model:(normalized count)")
print (bow_normalized)
print()

# TF-IDF Model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("TF-IDF Model:")
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())
print()


#  Word2Vec Embeddings

from gensim.models import Word2Vec

word2vec_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=10,   # embedding size
    window=2,
    min_count=1,
    sg=1              # Skip-gram
)

print("=== Word2Vec Embedding for 'learning' ===")
print(word2vec_model.wv["learning"])
print()


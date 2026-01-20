# Perform text cleaning, perform lemmatization (any method), remove stop words (any method), label encoding. Create representations using TF-IDF. Save outputs

import re 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

data ={
    "text": [
        "Cats are running faster than dogs",
        "The dog was barking loudly",
        "Students are studying machine learning",
        "She enjoys reading books and writing code"
    ],
    "label": ["animal", "animal", "education", "education"]
}

df = pd.DataFrame(data)

# cleaning

def clean_text(text):
    text = text.lower() 
    text = re.sub(r"[^a-z\s]", "", text)
    return text 

df['cleaned_text'] = df['text'].apply(clean_text)

# Lemmatization

# technique :- Rule base lemmatization

def simple_lemmatize(text):
    word= text.split()
    lemmas = []
    for w in word:
        if w.endswith('ing'):
            lemmas.append(w[:-3])
        elif w.endswith('ed'):
            lemmas.append(w[:-2])
        elif w.endswith('s')and len(w) > 3:
            lemmas.append(w[:-1])
        else:
            lemmas.append(w)
    return ' '.join(lemmas)

df['lemmatized_text'] = df['cleaned_text'].apply(simple_lemmatize)


# TF-IDF (use to count the frequency of words in the document)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['lemmatized_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

LabelEncoder=LabelEncoder()
df['encoded_label'] = LabelEncoder.fit_transform(df['label'])

# save outputs
df.to_csv("processed_text.csv", index=False)
tfidf_df.to_csv("tfidf_features.csv", index=False)

"Files saved successfully"


    

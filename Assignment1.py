import nltk
from nltk.tokenize import(
    WhitespaceTokenizer,
    wordpunct_tokenize,
    TreebankWordTokenizer,
    TweetTokenizer,
    MWETokenizer
)

from nltk.stem import PorterStemmer,  SnowballStemmer, WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

paragraph="""Natural language processing (NLP) is a subfield of artificial intelligence (AI) focused on the interaction between computers and humans through natural language. The ultimate objective of NLP is to enable computers to understand, interpret, and generate human languages in a way that is valuable."""

print("Original Paragraph:")
print(paragraph)
print("="*50)

# Tokenization Techniques 

# split text only on spaces
print("1. Whitespace Tokenizer:")
whitespace_tokenizer = WhitespaceTokenizer()
whitespace_tokens = whitespace_tokenizer.tokenize(paragraph)
print(whitespace_tokens)
print("="*50)

# seprates words and punctuation
print("2. WordPunct Tokenizer:")
wordpunct_tokens = wordpunct_tokenize(paragraph)
print(wordpunct_tokens)
print("="*50)

# Tokenizer for handling grammer rules
print("3. Treebank Word Tokenizer:")
treebank_tokenizer = TreebankWordTokenizer()
treebank_tokens = treebank_tokenizer.tokenize(paragraph)
print(treebank_tokens)
print("="*50)

# handle hashtags, mentions, and emoticons
print("4. Tweet Tokenizer:")
tweet_tokenizer = TweetTokenizer()
tweet_tokens = tweet_tokenizer.tokenize(paragraph)
print(tweet_tokens)
print("="*50)

# Multi-Word Expression Tokenizer
print("5. Multi-Word Expression Tokenizer:")
mwe_tokenizer = MWETokenizer([('natural', 'language', 'processing'), ('artificial', 'intelligence')])
mwe_tokens = mwe_tokenizer.tokenize(wordpunct_tokens)
print(mwe_tokens)
print("="*50)


# Stemming Techniques


tokens_for_stemming = [word.lower() for word in treebank_tokens if word.isalpha()]


# Produces root words but may not be meaningful
porter_stemmer = PorterStemmer()
porter_stems = [porter_stemmer.stem(word) for word in tokens_for_stemming]
print("6. Porter Stemming:")
print(porter_stems)
print("-" * 50)


# Improved and more accurate than Porter
snowball_stemmer = SnowballStemmer("english")
snowball_stems = [snowball_stemmer.stem(word) for word in tokens_for_stemming]
print("7. Snowball Stemming:")
print(snowball_stems)
print("-" * 50)

# Lemmatization Techniques

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens_for_stemming]
print("8. WordNet Lemmatization:")
print(lemmatized_words)
print("-" * 50)
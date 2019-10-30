# learning_coursera_nlp

## Basics


### Linguistics

Morphology, Syntax, Semantics and Pragmatics play an important role

## Text Pre-processing

* Examples: Use it for sentiment analysis
* What is text? --> It is a sequence of words
* What is a word? --> Meaningful sequence of characters
* *Tokenization* is the process that splits an input sequence into tokens
* Examples: WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer (nltk)
* *Stemming* is a process to remove/replace suffixes to get the root of the word, the stem
* Example: Porters stemmer (nltk)
* *Lemmatization* is the process of getting the dictionary form of the word
* Example: WordNet (nltk)
* Further examples: *Normalize capital letters*, *Acronyms* (can be solved by regular expressions)

### Transform Tokens into Words
* n-grams are a bag of words
* We can have too much n-grams so we have to remove some and keep the mediam frequency n-grams
* Term frequency (TF) is the frequency for term t
* Inverse document frequency (IDF) is the total number of documents in the corpus
* tfidf value can be reached by a high frequency and a low document frequency of the term in collection of documents
* Why is TF-IDF better than BOW:
  * We do not have to have so many n-grams

### Linear Model for sentiment analysis
* Examples to use: Baf of 1-grams with TF-IDF values ---> delivers extremely sparse matrix
* Better: Logistic regression
* Event better: Deep Learning

### SPAM Filtering
* Using hasing for feature identification works in practice
* Personalized feature hashing work best for this tasks
* Linear models - like bag of words - scale well for production
* Why doing spam classification for large dataset?
 * Because the simple linear classifier performs better
 * Examples that shows this statement: ad-click prediction

### Neural Net for words
* We create a sparse matrice with bag of words
* Neural Networks we create a dense representation
* We take the sum of word3vec vectors, and it can be a good feature!
* 1D convolution works even better as it analyzes also 2-grams


---> Stopped at 003/010

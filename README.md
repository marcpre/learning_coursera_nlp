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


---> Stopped at 002/007

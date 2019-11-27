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
* Examples to use: Bag-of-Words of 1-grams with TF-IDF values ---> delivers extremely sparse matrix
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
* If we need to train a neural network on characters we can use n-grams and 1D convolution
* For the final architecture we take only *1014* characters and apply 1D convolution and max pooling 6 times
* Apply Multi-Layer Perceptron on the dataset
* Deep-models work better for large datasets

### Language Models
* We can use n-grams to estimate the probability of the next word
* How to decode which model ist better:
 * Extrinsic evaluation: spelling correction etc.
 * Intrinsic evaluation: Hold-out perplexity (we held out data and test the model)

## Hidden Markov Model
* The Problem: Given a sequence of tokens and sequence of labels. Bring them together.
* Approaches:
 * Rule-based models
 * Seperate label classifiers for each token
 * Sequence models (HMM, MEMM, CRF)
 * Neural Networks
* Use Hidden Markov Models for this task
* How to apply the Hidden Markov model to our text?
* The problem of applying is: what is the most probable sequence of hidden states?
* Solution: Viterbi decoding

## Name Entity recognition
* Conditional random fields are models that are useful for graph representation
* For these models, we have to generate features to feed to the model
* We can use the method label-observation features

## Neural Language Models
* Curse of dimensionality --> If words are threaten through the model seperately
* Instead: Learn distributed representations for words --> Use Neural Network

# Recurrent Neural Network
* Tensorflow tutorial for recurrent neural network 
* Sequence tagging task is a good example for recurrent neural network

# Distributional semantics
* We need this in search to determine the most similar results
* Using "Positive Pointwise Mutual Information", we know if a word is random or not and if is independent
* Furthermore with PPMI we get only the value without 0
* Context are words by a sliding window
* Matrix factorization
 * Using singular value decomposition (SVD)
* Word2vec - 2 Arcitectures 
 * Continuous Bag-of-words
 * Continuous Skip-gram
* Wen can build...
* Word similarities - How can we evaluate the word similarities of word2vec?
  * Human judgement
  * Spearman's correlation
* Word analogies - How can we evaluate this task?
  * as above
* Doc2vec (gensim library)
 * We can evaluate it by a test dataset
* Challenges with word2vec:
 * Can have problems within accuracy, when words are not very similar
 * Has challenges with word analogies
* Sentence representation
 * Morphology can help to improve word embeddings
 * FastText model is a good example
 * StarSpace tried to build a general approach for learning of words or document embeddings
 * Deep Learning approaches like CNN, RNN (hierachical + sequence)
 * Skip-thought vectors: You want to predict the next sentence, hidden representation is called thought vector

# Topic Models
* Is an alternative way to build vector representation of texts
* Topics are described with words, It is just a probability distribution
* Topic Model:
 * PLSA - was developed in 1999
 * How to train PLSA?
 * Using EM Algorithm
* Zoo of Topic Models: 
* Latend Dirichlet Allocation (most popular topic model)
* New Path for new topic models is bayesian methods and graphical models
* Different extensions:
  * Hierachical topic models
  * Dynamic topic models - topics can evolve over time
  * Multilingual topic models

# Machine Translation
* Sources: Europarlament Translations, Book Translations, Wikipedia Translations etc.
* Problems with data: Noisy, Specific domain, Rare language pairs, not aligend, not enough
* Evaluation: How to compare two arbitrary translations?
* BLEU is a popular automatic technique
* Decouple the translation task into two models: Language model and Translation model
* Another idea: noisy chanel is the idea of "transforming" the source into the wanted sentence

# Word Alignment
* Using a word alignment matrix to visualize word alignment model

# Encoder-decoder attention architecture
* The task of the encoder is to build a hidden representation of an input
* Sometimes an encoder is also called thought/context vector
* Attentino mechanisn, 3 ways to calculate the weights (dot-product, weigths or via a neural network)
* Attention saves time
* How to implement conversational bots?
  * What is a chat bot?
    * Goal-oriented vs chit-chat bot
       
# Summarization and Simplification
* Summarization is a sequence to sequence task
  * Two Types, abstractive and extractive
* F.ex.: Textsum implementation from Google
* One of the most advanced model: Point generator network with coverage

# Task oriented dialog systems
* F.ex.: Amazon Alexa etc.
* "Intends" has to be pre-classified
* Intend is like a form a user has to fill in
* Intent classifier:
  * Any model on BOW with n-grams and TF-IDF
  * RNN
  * CNN
* Slot tagger:
  * Regex
  * Conditional random fields
  * RNN seq2seq
  * CNN seq2seq
  * Any seq2seq with attention




---> Stopped at 012/038

# Sentiment-Analysis

## Introduction 
Sentiment Analysis is useful for identifying trends of public opinion in the social media, for the
purpose of marketing and consumer research. It has its uses in getting customer
feedback about new product launches, political campaigns and even in financial
markets. It aims to determine the attitude of a speaker or a writer with respect to some
topic or simply the contextual polarity of a document. Early work in this area was
done by Turney and Pang who applied different methods for detecting the polarity of
product and movie reviews.
Sentiment analysis is a complicated problem but experiments have been done using
Naive Bayes, maximum entropy classifiers and support vector machines. Pang et al.
found the SVM to be the most accurate classifier.

## Data

The Rotten Tomatoes movie review dataset is a corpus of movie reviews used for
sentiment analysis, originally collected by Pang and Lee. In their work on sentiment
treebanks, Socher et al. used Amazon's Mechanical Turk to create fine-grained labels
for all parsed phrases in the corpus. This competition presents a chance to benchmark
your sentiment-analysis ideas on the Rotten Tomatoes dataset.

## Algorithms used

### Naive Bayes
Naive Bayes is a simple but surprisingly powerful algorithm for predictive modelling.
Quick introduction to Bayes’ theorem: In machine learning we are often interested in selecting the
best hypothesis (h) given data (d). In a classification problem, our hypothesis (h) may be the class to
assign for a new data instance (d).
One of the easiest ways of selecting the most probable hypothesis given the data that we have that
we can use as our prior knowledge about the problem. Bayes’ Theorem provides a way that we can
calculate the probability of a hypothesis given our prior knowledge.
**Bayes’ Theorem is stated as: P(h|d) = (P(d|h) * P(h)) / P(d)**

Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of
variables (features/predictors) in a learning problem. Maximum-likelihood training can be done by
evaluating a closed-form expression, which takes linear time, rather than by expensive iterative
approximation as used for many other types of classifiers.

### Support Vector Machine

“Support Vector Machine” (SVM) is a supervised machine learning algorithm which can be
used for both classification or regression challenges. However, it is mostly used in classification
problems. In this algorithm, we plot each data item as a point in n-dimensional space (where n is
number of features you have) with the value of each feature being the value of a particular
coordinate. Then, we perform classification by finding the hyper-plane that differentiate the two
classes very well.
Support Vectors are simply the co-ordinates of individual observation. Support Vector Machine is
a frontier which best segregates the two classes (hyper-plane/ line).

## FEATURE SELECTION AND ENGINEERING

### STOPWORDS
In computing, stop words are words which are filtered out before or after processing of
natural language data. Though "stop words" usually refers to the most common words in a
language, there is no single universal list of stop words used by all natural language
processing tools, and indeed not all tools even use such a list.
We ignored the most common and least common words appearing in the corpus as they did
not contribute to any sentimental value in Python using stopwords from the nltk library.

**stop_words = stopwords.words('English')**

### STEMMER
o  Stemmers remove morphological affixes from words, leaving only the word stem.\
o  Stemming algorithms are used to find the “root word” or stem of a given word. We have
used the Porter Stemmer.\
o   We also used a stemmer from the PorterStemmer library to stem words to obtain the root of
the word.

### VECTORIZER
The sklearn.feature_extraction module deals with feature extraction from raw data. It
currently includes methods to extract features from text and images. Among that
feature_extraction.text.TfidfVectorizer([…]) which
Convert a collection of raw documents to a matrix of TF-IDF features.
We also tuned the parameters (max_df, min_df) in the TfidfVectorizer to ignore words
appearing in more than 90% of the corpus and less than 1% of the corpus.

**vectorizer = TfidfVectorizer(stop_words = stop_words, max_df =
0.9, min_df = 0.01, ngram_range = [1,2], analyzer =
stemmed_words)**

### Training 

svm_clf =svm.LinearSVC(C=0.1)\
vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])\
vec_clf.fit(X_train,y_train)

### Output

![image](https://user-images.githubusercontent.com/32728058/119485992-5c29d200-bd75-11eb-9483-512a6a3fc42a.png)


## Conclusion

The Naïve Bayes model gives us an accuracy of 72.39 % and SVM model gives an accuracy of roughly 82% in the on the test data, which is better than our Naïve
Bayes Model.

![image](https://user-images.githubusercontent.com/32728058/119485372-9a72c180-bd74-11eb-97d8-622e7c448c4c.png)



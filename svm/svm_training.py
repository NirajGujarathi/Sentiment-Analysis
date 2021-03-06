import csv
import os
import re
import nltk
import scipy
import sklearn.metrics
import sentiment
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

#Generating the Training and testing vectors

def getTrainingAndTestData():
        X = []
        y = []

        #Training data 1: Sentiment 140
        f=open(r'training_test.csv','r')
        reader = csv.reader(f)

        for row in reader:
            X.append(row[5])
            y.append(1 if (row[0]=='4') else 0)
            
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X,y,test_size=0.20, random_state=42)
        return X_train, X_test, y_train, y_test

#Process Tweets (Stemming+Pre-processing)

def processTweets(X_train, X_test):
        X_train = [sentiment.stem(sentiment.preprocessTweets(tweet)) for tweet in X_train]
        X_test = [sentiment.stem(sentiment.preprocessTweets(tweet)) for tweet in X_test]
        return X_train,X_test
        
# SVM classifier

def classifier(X_train,y_train):
        stop_words = stopwords.words('English')
        vec = TfidfVectorizer(stop_words = stop_words, min_df=5, max_df=0.95, sublinear_tf = True,ngram_range=(1, 2))
        svm_clf =svm.LinearSVC(C=0.1)
        vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
        vec_clf.fit(X_train,y_train)
        joblib.dump(vec_clf, 'svmClassifier.pkl', compress=3)
        return vec_clf

# Main function

def main():
        X_train, X_test, y_train, y_test = getTrainingAndTestData()
        X_train, X_test = processTweets(X_train, X_test)
        vec_clf = classifier(X_train,y_train)
        y_pred = vec_clf.predict(X_test)
        print(sklearn.metrics.classification_report(y_test, y_pred))  
        
if __name__ == "__main__":
    main()

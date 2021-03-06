import sys
import time
import re
import nltk
from sklearn.externals import joblib

#Processing Tweets

def preprocessTweets(tweet):
	
	#Convert www.* or https?://* to URL
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
	
	#Convert @username to __HANDLE
	tweet = re.sub('@[^\s]+','__HANDLE',tweet)  
	
	#Replace #word with word
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	
	#trim
	tweet = tweet.strip('\'"')
	
	# Repeating words like happyyyyyyyy
	rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
	tweet = rpt_regex.sub(r"\1\1", tweet)
	
	#Emoticons
	emoticons = \
	[
	 ('__positive__',[ ':-)', ':)', '(:', '(-:', \
					   ':-D', ':D', 'X-D', 'XD', 'xD', \
					   '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ] ),\
	 ('__negative__', [':-(', ':(', '(:', '(-:', ':,(',\
					   ':\'(', ':"(', ':((', ] ),\
	]

	def replace_parenth(arr):
	   return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]
	
	def regex_join(arr):
		return '(' + '|'.join( arr ) + ')'

	emoticons_regex = [ (repl, re.compile(regex_join(replace_parenth(regx))) ) \
			for (repl, regx) in emoticons ]
	
	for (repl, regx) in emoticons_regex :
		tweet = re.sub(regx, ' '+repl+' ', tweet)

	 #Convert to lower case
	tweet = tweet.lower()
	
	return tweet

#Stemming of Tweets

def stem(tweet):
		stemmer = nltk.stem.PorterStemmer()
		tweet_stem = ''
		words = [word if(word[0:2]=='__') else word.lower() \
					for word in tweet.split() \
					if len(word) >= 3]
		words = [stemmer.stem(w) for w in words] 
		tweet_stem = ' '.join(words)
		return tweet_stem


#Predict the sentiment

def predict(tweet,classifier):
	
	tweet_processed = stem(preprocessTweets(tweet))
			 
	if ( ('__positive__') in (tweet_processed)):
		 sentiment  = 'positive'
		
	elif ( ('__negative__') in (tweet_processed)):
		 sentiment  = 'negative'

	else:
		X =  [tweet_processed]
		sentiment = classifier.predict(X)
		sentiment = sentiment[0]
		if sentiment == 0:
			sentiment = 'negative'
		else:
			sentiment = 'positive'
	
	return sentiment
 

def main():
	print('Loading the Classifier, please wait....')
	classifier = joblib.load('svm.pkl')
	print('READY')

	while True:
		print(predict(input("Enter the text:"),classifier))
				
			
if __name__ == "__main__":
	main()
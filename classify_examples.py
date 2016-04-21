import pickle
import csv, codecs, string
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, DecisionTreeClassifier
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramAssocMeasures

def read_input(infile, classifier):
	with codecs.open(infile, 'rb') as csvfile:
		n=0
		time.clock()
		reader = csv.reader(csvfile)
		tokenizer = TweetTokenizer(preserve_case=True)
		for line in reader:
			n+=1
			text = tokenizer.tokenize(line[5].decode("utf-8"))
			text = [token for token in text if token != u'\ufffd']
			sent = classifier.classify(text)
			print line[5], sent
		print n + "Lines read in" + time.clock()

classifier = pickle.load(open("sentiment_classifier.pkl", 'rb'))
read_input("test_manual.csv", classifier)
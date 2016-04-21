# coding=UTF-8

import csv, codecs, string
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, DecisionTreeClassifier
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramAssocMeasures

def read_input(infile, NUM_TRAIN, NUM_TEST):
	with codecs.open(infile, 'rb') as csvfile:
		train = []
		test = []
		pos_tweets = 0
		neg_tweets = 0
		reader = csv.reader(csvfile)
		tokenizer = TweetTokenizer(preserve_case=True)
		for line in reader:
			if line[0] == "0":
				sent="Negative"
				neg_tweets+=1
				if neg_tweets < NUM_TRAIN:
					text = tokenizer.tokenize(line[5].decode("utf-8"))
					text = [token for token in text if token != u'\ufffd']
					train.append((text, sent))

				elif neg_tweets < NUM_TRAIN+NUM_TEST:
					text = tokenizer.tokenize(line[5].decode("utf-8"))
					text = [token for token in text if token != u'\ufffd']
					test.append((text, sent))

		
			if line[0] == "4":
				sent = "Positive"
				pos_tweets+=1
				if pos_tweets < NUM_TRAIN:
					text = tokenizer.tokenize(line[5].decode("utf-8"))
					text = [token for token in text if token != u'\ufffd']
					train.append((text, sent))
				elif pos_tweets < NUM_TRAIN+NUM_TEST:			
					text = tokenizer.tokenize(line[5].decode("utf-8"))
					text = [token for token in text if token != u'\ufffd']
					test.append((text, sent))
			

		return test, train



# Read in annotated data
NUM_TRAIN = 10000
NUM_TEST = 2500
test, train = read_input("train.csv",NUM_TRAIN,NUM_TEST)


sentiment_analyzer = SentimentAnalyzer()
#all_words = sentiment_analyzer.all_words([mark_negation(doc[0]) for doc in train])
all_words = sentiment_analyzer.all_words([doc[0] for doc in train])
unigrams = sentiment_analyzer.unigram_word_feats(all_words, min_freq=4)
# print unigrams
sentiment_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigrams)

training_set=sentiment_analyzer.apply_features(train)
test_set=sentiment_analyzer.apply_features(test)

trainer = NaiveBayesClassifier.train
classifier = sentiment_analyzer.train(trainer, training_set)
save_file(sentiment_analyzer, "sentiment_classifier.pkl")
for key,value in sorted(sentiment_analyzer.evaluate(test_set).items()):
	print("{0}: {1}".format(key,value))

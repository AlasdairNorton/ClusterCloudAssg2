# coding=UTF-8

import csv, codecs, string, re
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples


def read_input(infile, NUM_TRAIN, NUM_TEST):
	train = []
	test = []
	pos_tweets = 0
	neg_tweets = 0
	for line in twitter_samples.tokenized("positive_tweets.json"):
		sent = "Positive"
		#Remove usernames, urls
		for i,token in enumerate(line):
			
			line[i] = re.sub("@[\S]+", "USERNAME", line[i])
			line[i] = re.sub("www.[\S]+|https://[\S]+|http://[\S]+", "URL", line[i])
			newstr = ""
			for ch in line[i]:
				if ord(ch)>128:
					newstr+= "EMOJI_{0}".format(ord(ch))
					#print [ch], ord(ch)
				else:
					newstr+=(ch)
			line[i] = newstr

		pos_tweets+=1
		if pos_tweets < NUM_TRAIN:
			train.append((line, sent))
		else:			
			test.append((line, sent))			


	for line in twitter_samples.tokenized("negative_tweets.json"):
		sent = "Negative"
		neg_tweets+=1
		#Remove usernames, urls
		for i,token in enumerate(line):

			line[i] = re.sub("@[\S]+", "USERNAME", line[i])
			line[i] = re.sub("www.[\S]+|https://[\S]+", "URL", line[i])
			newstr = ""
			for ch in line[i]:
				if ord(ch)>128:
					newstr+= "EMOJI_{0}".format(ord(ch))
					#print [ch], ord(ch)
				else:
					newstr+=(ch)
			line[i] = newstr
		if neg_tweets < NUM_TRAIN:
			train.append((line, sent))
		else:		
			test.append((line, sent))	
	return test, train

def get_unigrams(document, unigrams):
	ret = {}
	for ung in unigrams:
		if ung in document:
			ret["contains({0})".format(ung)] = True
		else:
			ret["contains({0})".format(ung)] = False
	return ret


def get_test(infile, NUM_TEST):
	with codecs.open(infile, 'rb') as csvfile:
		test = []
		pos_tweets = 0
		neg_tweets = 0
		reader = csv.reader(csvfile)
		tokenizer = TweetTokenizer(preserve_case=True)
		for line in reader:
			if line[0] == "0":
				sent="Negative"
				neg_tweets+=1

				if neg_tweets < NUM_TEST:
					text = tokenizer.tokenize(line[5].decode("utf-8"))
					for i,token in enumerate(text):
						text[i] = re.sub("@[\S]+", "USERNAME", text[i])
						text[i] = re.sub("www.[\S]+|https://[\S]+", "URL", text[i])
						newstr = ""
						for ch in text[i]:
							if ord(ch)>128:
								newstr+= "EMOJI_{0}".format(ord(ch))
								#print [ch], ord(ch)
							else:
								newstr+=(ch)
						text[i] = newstr
					test.append((text, sent))

		
			if line[0] == "4":
				sent = "Positive"
				pos_tweets+=1
				
				if pos_tweets < NUM_TEST:			
					text = tokenizer.tokenize(line[5].decode("utf-8"))
					for i,token in enumerate(text):
						text[i] = re.sub("@[\S]+", "USERNAME", text[i])
						text[i] = re.sub("www.[\S]+|https://[\S]+", "URL", text[i])
						newstr = ""
						for ch in text[i]:
							if ord(ch)>128:
								newstr+= "EMOJI_{0}".format(ord(ch))
								#print [ch], ord(ch)
							else:
								newstr+=(ch)
						text[i] = newstr
					test.append((text, sent))
			

		return test


# Read in annotated data
NUM_TRAIN = 5000
NUM_TEST = 2000
test, train = read_input("train.csv",NUM_TRAIN,NUM_TEST)
test = get_test("train.csv", NUM_TEST)
print len(test), len(train)
sentiment_analyzer = SentimentAnalyzer()
all_words = sentiment_analyzer.all_words([doc[0] for doc in train])

# # Get list of terms+frequencies
# words_freqs = {}
# for tweet in train:
# 	for token in tweet[0]:
# 		if token in words_freqs:
# 			words_freqs[token] += 1
# 		else:
# 			words_freqs[token] = 1

# unigrams = [token for token in words_freqs if words_freqs[token] >= 4]
unigrams = sentiment_analyzer.unigram_word_feats(all_words, min_freq=4)
#bigrams = sentiment_analyzer.bigram_collocation_feats([doc[0] for doc in train], top_n=1000)
# print unigrams

sentiment_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigrams)
#sentiment_analyzer.add_feat_extractor(extract_bigram_feats, bigrams=bigrams)

training_set=sentiment_analyzer.apply_features(train)
test_set=sentiment_analyzer.apply_features(test)
#print training_set[0]
trainer = NaiveBayesClassifier.train
classifier = sentiment_analyzer.train(trainer, training_set)
save_file(sentiment_analyzer, "sentiment_classifier.pkl")
for key,value in sorted(sentiment_analyzer.evaluate(test_set).items()):
	print("{0}: {1}".format(key,value))
print test[0], sentiment_analyzer.classify(test[0][0])

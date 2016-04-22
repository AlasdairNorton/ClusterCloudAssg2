import pickle
import csv, codecs, string, json
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, DecisionTreeClassifier
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramAssocMeasures

def read_input(infile, outfile, classifier):
	with codecs.open(infile, 'rb') as infile:
		outfile = open(outfile, "w")
		n=0
		time.clock()
		tokenizer = TweetTokenizer(preserve_case=True)
		for line in infile:
			try:
				json_line = json.loads(line)
			except ValueError:
				# Sometimes two tweets get stuck on one line in the input files. Ignore these lines
				continue
			text = json_line["text"]
			n+=1
			text = tokenizer.tokenize(text)
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
			feats = classifier.apply_features(text)
			sent = classifier.classifier.prob_classify(feats[0])
			s = classifier.classify(text)
			l = line[:-2]+', "Sentiment": "'+s+'"}'
			#l = line[:-2]+", 'Sentiment': '"+s+"'}"
			#json_line["sentiment"] = s
			outfile.write(l+'\n')

			#print s, "\t", sent.prob("Positive"), "\t", sent.prob("Negative"),"\t", " ".join(text)
		print n, "lines read in", time.clock()

classifier = pickle.load(open("sentiment_classifier.pkl", 'rb'))
read_input("tweets_20160417121932.json", "outfile.json", classifier)

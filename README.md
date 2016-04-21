# ClusterCloudAssg2
Cluster&amp;Cloud Group 22

A git repo is required for the project, so I thought I'd start us off by uploading my sentiment analysis code
Contents:
trainer.py - trains a classifier and saves to sentiment_classifier.pkl
classify_instances.py - loads the classifier from pickle and runs it on tweet data. Currently runs on an input csv and outputs classifications to stdout; should be changed to take data in the format given by the twitter crawlers and modify the files to add the classification
train.csv - twitter training data -sourced from http://help.sentiment140.com/for-students/

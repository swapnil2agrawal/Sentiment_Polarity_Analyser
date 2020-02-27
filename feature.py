"""
File Name : feature.py
Author : Swapnil_Agrawal
Date Created : 01/23/2019
Python Version : 3.6
Detail : Extract features from raw file to form feature vector and in usable form for logistic regression 
Input : python feature.py train_data.tsv valid_data.tsv test_data.tsv \
dict.txt formatted_train.tsv formatted_valid.tsv formatted_test.tsv 1
"""

import numpy as np 
import sys
import csv

if __name__ == '__main__':
	dict = {}
	# 4 is the dictionary file containing all the words 
	reader = csv.reader(open(sys.argv[4], "r"), delimiter='\t')


def model(nr, nw, model_num):

	"""nr = argument number of the file from which features need to be extracted
	nw = argument number of the file to which extractd features are stored 
	model_num = 1/2 = Which model to use for feature extraction	"""

	reader = csv.reader(open(sys.argv[nr], "r"), delimiter='\t') 
	ex = np.array(list(reader))

	label = np.zeros(len(ex))

	writer = open(sys.argv[nw], "w")

	if(model_num==1):

		# model - 1 = indicating which word occurs atleast once in the review from the dictionary

		for i in range(len(ex)):

			# noting down the label assigned to each movie from review
			label[i] = ex[i][0]

			writer.write(str(int(label[i])))
			
			A = ex[i][1].split()
			
			# remove repetitive
			x = {} #empty dictionary to keep a track of repetitive words

			for a in A:				
				if(a in dict and dict[a] not in x):	
					# adding a label 1 to words which are occuring in the review
				 	x[dict[a]]=1
				 	writer.write("\t")
				 	writer.write(dict[a]+":1")
				 	
			writer.write("\n")

	elif(model_num==2):

		"""model - 2 = keeping a count of all the words in review and removing the words which are 
		occuring more than 4 times as they may be just punctuation"""

		for i in range(len(ex)):
			# noting down the label assigned to each movie from review
			label[i] = ex[i][0]

			writer.write(str(int(label[i])))
			
			A = ex[i][1].split()
			
			x = {} #dictionary to keep track of words occuring in review

			for a in A:
				
				if(a in dict):
					# if word already there, add a count or else add it to the dictionary

			 		if(dict[a] in x):
			 			x[dict[a]]=x[dict[a]]+1

			 		elif(dict[a] not in x):
			 			x[dict[a]]=1
			
			# to remove words which are occuring more than 4 times
			y = {}
			for a in A:
				if(a in dict and x[dict[a]] < 4 and dict[a] not in y):	
					y[dict[a]]=1
					writer.write("\t")
					writer.write(dict[a]+":1")
					
			writer.write("\n")


def main():
	
	D = np.array(list(reader))
	
	for d in D:
		## storing in dictionary
		a, b = d[0].split()
		dict[a] = b

	model(1, 5, int(sys.argv[8]))  #1 and 5 represent the training input and output files
	model(2, 6, int(sys.argv[8]))  #2 and 6 represent the validation input and output files
	model(3, 7, int(sys.argv[8]))  #3 and 7 represent the test input and output files


main()
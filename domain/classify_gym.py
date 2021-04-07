
import logging
import numpy as np
import sys
import math
import csv

class ClassifyEnv():

	def __init__(self, trainSet, target):
		"""
		Data set is a tuple of
		[0] input data: [nSamples x nInputs]
		[1] labels:     [nSamples x 1]

		Example data sets are given at the end of this file
		"""

		self.t = 0          # Current batch number
		self.t_limit = 0    # Number of batches if you want to use them (we didn't)
		self.batch   = 1000 # Number of images per batch

		self.trainSet = trainSet
		self.target   = target

		self.state = None
		self.trainOrder = None
		self.currIndx = None

	def reset(self):
		''' Initialize State'''
		# print('Lucky number', np.random.randint(10)) # same randomness?
		self.trainOrder = np.random.permutation(len(self.target))
		self.t = 0 # timestep
		self.currIndx = self.trainOrder[self.t:self.t+self.batch]
		self.state = self.trainSet[self.currIndx,:]
		return self.state

	def step(self, action):
		'''
		Judge Classification, increment to next batch
		action - [batch x output] - softmax output
		'''
		y = self.target[self.currIndx]
		m = y.shape[0]

		log_likelihood = -np.log(action[range(m),y])
		loss = np.sum(log_likelihood) / m
		reward = -loss

		if self.t_limit > 0: # We are doing batches
			reward *= (1/self.t_limit) # average
			self.t += 1
			done = False
			if self.t >= self.t_limit:
				done = True
			self.currIndx = self.trainOrder[
				(self.t*self.batch):(self.t*self.batch + self.batch)]

			self.state = self.trainSet[self.currIndx,:]
		else:
			done = True

		obs = self.state
		return obs, reward, done, {}

	def accuracy(self, action):
		p = np.argmax(action, axis=1)
		y = self.target[self.currIndx]
		accuracy = (float(np.sum(p==y)) / self.batch)
		return accuracy

# -- Data Sets ----------------------------------------------------------- -- #

def mnist_256():
	import os
	'''
	Converts 28x28 mnist digits to [16x16]
	[samples x pixels]  ([N X 256])
	'''
	with open(os.path.abspath(os.path.dirname(__file__)) + "/mnist/data/train_img.csv") as f:
		reader = csv.reader(f, delimiter = " ")
		img = np.array([[float(v) for v in row] for row in reader])
	with open(os.path.abspath(os.path.dirname(__file__)) + "/mnist/data/train_label.csv") as f:
		reader = csv.reader(f, delimiter = " ")
		label = np.array([int(row[0]) for row in reader])
	return img, label

def mnist_256_test():
	import os
	'''
	Converts 28x28 mnist digits to [16x16]
	[samples x pixels]  ([N X 256])
	'''
	with open(os.path.abspath(os.path.dirname(__file__)) + "/mnist/data/test_img.csv") as f:
		reader = csv.reader(f, delimiter = " ")
		img = np.array([[float(v) for v in row] for row in reader])
	with open(os.path.abspath(os.path.dirname(__file__)) + "/mnist/data/test_label.csv") as f:
		reader = csv.reader(f, delimiter = " ")
		label = np.array([int(row[0]) for row in reader])
	return img, label

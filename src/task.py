import numpy as np
from domain import make_env, make_test_env
from .ind import act, selectAct

class Task():
	"""Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
	"""
	def __init__(self, game, paramOnly=False, nReps=1):
		"""Initializes task environment

		Args:
			game - (string) - dict key of task to be solved (see domain/config.py)

		Optional:
			paramOnly - (bool)  - only load parameters instead of launching task?
			nReps     - (nReps) - number of trials to get average fitness
		"""
		# Network properties
		self.nInput   = game.input_size
		self.nOutput  = game.output_size
		self.actRange = game.h_act
		self.absWCap  = game.weightCap
		self.layers   = game.layers
		self.activations = np.r_[np.full(1,1), game.i_act, game.o_act]

		# Environment
		self.maxEpisodeLength = game.max_episode_length
		self.actSelect = game.actionSelect

		if not paramOnly:
			self.env = make_env(game.env_name)
			self.test_env = make_test_env(game.env_name)

		# Special needs...
		self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))

	def calcReward(self, wVec, aVec, seed=-1):
		"""Evaluate individual on task
		Args:
			wVec    - (np_array) - weight matrix as a flattened vector
					[N**2 X 1]
			aVec    - (np_array) - activation function of each node
					[N X 1]    - stored as ints (see applyAct in ann.py)

		Optional:
			view    - (bool)     - view trial?
			seed    - (int)      - starting random seed for trials

		Returns:
			fitness - (float)    - reward earned in trial
		"""
		if seed >= 0:
			random.seed(seed)
			np.random.seed(seed)
			self.env.seed(seed)

		state = self.env.reset()
		self.env.t = 0

		annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
		action = selectAct(annOut,self.actSelect)

		state, reward, done, info = self.env.step(action)
		if self.maxEpisodeLength == 0:
			return reward
		else:
			totalReward = reward

		for tStep in range(self.maxEpisodeLength):
			annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
			action = selectAct(annOut,self.actSelect)
			state, reward, done, info = self.env.step(action)
			totalReward += reward
			if done:
				break

		return totalReward

	def calcAccuracy(self, wVec, aVec, test):
		if test:
			state = self.test_env.reset()
			self.test_env.t = 0
			annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
			action = selectAct(annOut, self.actSelect)
			return self.test_env.accuracy(action)
		else:
			state = self.env.reset()
			self.env.t = 0
			annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
			action = selectAct(annOut, self.actSelect)
			return self.env.accuracy(action)

	def setWeights(self, wVec, wVal):
		"""Set single shared weight of network

		Args:
			wVec    - (np_array) - weight matrix as a flattened vector
					[N**2 X 1]
			wVal    - (float)    - value to assign to all weights

		Returns:
			wMat    - (np_array) - weight matrix with single shared weight
					[N X N]
		"""
		# Create connection matrix
		wVec[np.isnan(wVec)] = 0
		dim = int(np.sqrt(np.shape(wVec)[0]))
		cMat = np.reshape(wVec,(dim,dim))
		cMat[cMat!=0] = 1.0

		# Assign value to all weights
		wMat = np.copy(cMat) * wVal
		return wMat

	def getFitness(self, hyp, wVec, aVec, \
					seed=-1,nRep=False,returnVals=False):
		"""Get fitness of a single individual with distribution of weights

		Args:
			wVec    - (np_array) - weight matrix as a flattened vector
					[N**2 X 1]
			aVec    - (np_array) - activation function of each node
					[N X 1]    - stored as ints (see applyAct in ann.py)
			hyp     - (dict)     - hyperparameters
			['alg_wDist']        - weight distribution  [standard;fixed;linspace]
			['alg_absWCap']      - absolute value of highest weight for linspace

		Optional:
			seed    - (int)      - starting random seed for trials
			nReps   - (int)      - number of trials to get average fitness


		Returns:
			fitness - (float)    - mean reward over all trials
		"""

		if nRep is False:
			nRep = hyp['alg_nReps']
		nVals = hyp['alg_nVals']
		wVals = self.get_wVals(hyp['alg_wDist'], nVals)

		# Get reward from 'reps' rollouts -- test population on same seeds
		reward = np.empty((nRep,nVals))
		for iRep in range(nRep):
			for iVal in range(nVals):
				wMat = self.setWeights(wVec, wVals[iVal])
				if seed == -1:
					reward[iRep,iVal] = self.calcReward(wMat, aVec, seed=seed)
				else:
					reward[iRep,iVal] = self.calcReward(wMat, aVec, seed=seed+iRep)

		if returnVals is True:
			return np.mean(reward,axis=0), wVals
		return np.mean(reward,axis=0)

	def getAcculacy(self, hyp, wVec, aVec, test, seed=None):
		if seed is not None:
			np.random.seed(seed)

		nVals = hyp['alg_nVals']
		wVals = self.get_wVals(hyp['alg_wDist'], nVals)

		acculacy = np.empty(nVals)
		for iVal in range(nVals):
			wMat = self.setWeights(wVec, wVals[iVal])
			acculacy[iVal] = self.calcAccuracy(wMat, aVec, test)

		return acculacy

	def get_wVals(self, alg_wDist, alg_nVals):
		# Set weight values to test WANN with
		if (alg_wDist == "standard") and alg_nVals == 6: # Double, constant, and half signal
			return np.array((-2.0, -1.0, -0.5, 0.5, 1.0, 2.0))
		if (alg_wDist == "standard") and alg_nVals == 2:
			return np.array((-2.0, 2.0))
		elif (alg_wDist == "positive" and alg_nVals == 6):
			return np.array((0.5, 0.75, 1.0, 1.25, 1.5, 2.0))
		elif (alg_wDist == "positive" and alg_nVals == 3):
			return np.array((0.5, 1.0, 2.0))
		elif (alg_wDist == "positive" and alg_nVals == 1):
			return np.array((1.0))
		elif (alg_wDist == "random"):
			wVals = np.random.rand(alg_nVals) * 3 - 1.5
			return [w - 0.5 if w < 0 else w + 0.5 for w in wVals]
		elif (alg_wDist == "symmetricrandom"):
			wVals = np.random.rand(alg_nVals // 2) * 1.5 + 0.5
			return np.append(wVals, -wVals)
		else:
			return np.linspace(-self.absWCap, self.absWCap, alg_nVals)

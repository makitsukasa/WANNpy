import copy
import numpy as np
from .ind import Ind
from .task import Task
from domain import games
from .nsga_sort import nsga_sort

class Wann():
	"""WANN main class. Evolves population given fitness values of individuals.
	"""

	''' Subfunctions '''
	from .variation import evolvePop, recombine, crossover,	mutAddNode, mutAddConn, topoMutate

	def __init__(self, hyp):
		"""Intialize WANN algorithm with hyperparameters
		Args:
			hyp - (dict) - algorithm hyperparameters

		Attributes:
			p       - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
			pop     - (Ind)      - Current population
			species - (Species)  - Current species
			innov   - (np_array) - innovation record
					[5 X nUniqueGenes]
					[0,:] == Innovation Number
					[1,:] == Source
					[2,:] == Destination
					[3,:] == New Node?
					[4,:] == Generation evolved
			gen     - (int)      - Current generation
		"""
		self.p = hyp       # Hyperparameters
		self.pop = []      # Current population
		self.species = []  # Current species
		self.innov = []    # Innovation number (gene Id)
		self.gen = 0
		global games
		self.task = Task(games[hyp["task"]])

	def train(self):
		seed_a = np.random.randint(4294967296, dtype="int64")
		for gen in range(self.p['maxGen']):
			if len(self.pop) == 0:
				self.initPop()        # Initialize population
			else:
				self.probMoo()        # Rank population according to objectives
				self.evolvePop()      # Create child population

			self.evaluate()           # Send pop to evaluate

			f = sorted(self.pop, reverse=True, key=lambda i: i.fitness)[0].fitness
			if gen + 1 in [1, 10, 20, 30, 40, 50, 70, 4096, self.p['maxGen']] + list(range(100, 10000, 50)):
				# a = self.acculacy(seed_a)
				a = self.acculacy_all()
				print("gen {} | fitness: {} acculacy: {}".format(gen + 1, f, a))
			else:
				print("gen {} | fitness: {}".format(gen + 1, f))

	def initPop(self):
		"""Initialize population with a list of random individuals
		"""
		#  Create base individual

		# - Create Nodes -
		nodeId = np.arange(0,self.p['ann_nInput']+ self.p['ann_nOutput']+1,1)
		node = np.empty((3,len(nodeId)))
		node[0,:] = nodeId

		# Node types: [1:input, 2:hidden, 3:bias, 4:output]
		node[1,0]             = 4 # Bias
		node[1,1:self.p['ann_nInput']+1] = 1 # Input Nodes
		node[1,(self.p['ann_nInput']+1):\
			(self.p['ann_nInput']+self.p['ann_nOutput']+1)]  = 2 # Output Nodes

		# Node Activations
		node[2,:] = self.p['ann_initAct']

		# - Create Conns -
		nConn = (self.p['ann_nInput']+1) * self.p['ann_nOutput']
		ins   = np.arange(0,self.p['ann_nInput']+1,1)            # Input and Bias Ids
		outs  = (self.p['ann_nInput']+1) + np.arange(0,self.p['ann_nOutput']) # Output Ids

		conn = np.empty((5,nConn,))
		conn[0,:] = np.arange(0,nConn,1)    # Connection Id
		conn[1,:] = np.tile(ins, len(outs)) # Source Nodes
		conn[2,:] = np.tile(outs,len(ins) ) # Destination Nodes
		conn[3,:] = 1                       # Weight Value
		conn[4,:] = 1                       # Enabled?

		# Create population of individuals (for WANN weight value doesn't matter)
		pop = []
		for i in range(self.p['popSize']):
			newInd = Ind(conn, node)
			newInd.conn[4,:] = np.random.rand(1, nConn) < self.p['prob_initEnable']
			newInd.express()
			newInd.birth = 0
			pop.append(copy.deepcopy(newInd))

		# - Create Innovation Record -
		innov = np.zeros([5,nConn])
		innov[0:3,:] = pop[0].conn[0:3,:]
		innov[3,:] = -1

		self.pop = pop
		self.innov = innov

	def probMoo(self):
		"""Rank population according to Pareto dominance.
		"""

		# Alternate second objective
		if self.p['alg_probMoo'] < np.random.rand():
			# Compile objectives
			meanFit = np.asarray([ind.fitness for ind in self.pop])
			maxFit  = np.asarray([ind.fitMax  for ind in self.pop])
			objVals = np.c_[meanFit, maxFit]
			rank = nsga_sort(objVals)
		else:
			# Compile objectives
			meanFit = np.asarray([ind.fitness for ind in self.pop])
			nConns  = np.asarray([ind.nConn   for ind in self.pop])
			nConns[nConns==0] = 1 # No conns is always pareto optimal (but boring)
			objVals = np.c_[meanFit, 1/nConns] # Maximize
			rank = nsga_sort(objVals)

		# Assign ranks
		for i in range(len(self.pop)):
			self.pop[i].rank = rank[i]

	def evaluate(self):
		"""Sends population to workers for evaluation one batch at a time.

		Args:
			pop - [Ind] - list of individuals
			.wMat - (np_array) - weight matrix of network
					[N X N]
			.aVec - (np_array) - activation function of each node
					[N X 1]

		Return:
			reward  - (np_array) - fitness value of each individual
					[N X 1]

		Todo:
			* Asynchronous evaluation instead of batches
		"""

		for i in range(len(self.pop)):
			wVec   = self.pop[i].wMat.flatten()
			aVec   = self.pop[i].aVec.flatten()
			result = self.task.getFitness(self.p, wVec, aVec)
			self.pop[i].fitness = np.mean(result)
			self.pop[i].fitMax = np.max(result)

	def acculacy(self, seed = None):
		ind = sorted(self.pop, key=lambda i: i.fitness, reverse=True)[0]
		wVec = ind.wMat.flatten()
		return np.mean(self.task.getAcculacy(self.p, wVec, ind.aVec, False, seed)),\
			np.mean(self.task.getAcculacy(self.p, wVec, ind.aVec, True, seed))

	def acculacy_all(self, seed = None):
		ind = sorted(self.pop, key=lambda i: i.fitness, reverse=True)[0]
		wVec = ind.wMat.flatten()
		return self.task.getAcculacy(self.p, wVec, ind.aVec, False, seed),\
			self.task.getAcculacy(self.p, wVec, ind.aVec, True, seed)

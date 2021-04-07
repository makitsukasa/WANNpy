import numpy as np
import src
import domain
from src.nsga_sort import nsga_sort

hyp = {
	"task":"mnist256",
	"alg_wDist": "standard",
	"alg_nVals": 6,
	# "maxGen": 70 * 6 // 6,
	"maxGen": 4096 * 6 // 6,
	"popSize": 960,
	"alg_nReps": 1,
	"alg_probMoo": 0.80,
	"prob_crossover": 0.0,
	"prob_mutAct":  0.50,
	"prob_addNode": 0.25,
	"prob_addConn": 0.20,
	"prob_enable":  0.05,
	"prob_initEnable": 0.05,
	"select_cullRatio": 0.2,
	"select_eliteRatio": 0.2,
	"select_tournSize": 32,
	"save_mod": 8,
	"bestReps": 20,
	"alg_nMean": 8,
	"ann_nInput": 16**2,
	"ann_nOutput": 10,
	"ann_initAct": 0,
	"ann_actRange": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

class Wann_mean_nconn(src.wann.Wann):
	def probMoo(self):
		"""Rank population according to Pareto dominance.
		"""
		# Alternate second objective
		# Compile objectives
		meanFit = np.asarray([ind.fitness for ind in self.pop])
		nConns  = np.asarray([ind.nConn   for ind in self.pop])
		nConns[nConns==0] = 1 # No conns is always pareto optimal (but boring)
		objVals = np.c_[meanFit, 1/nConns] # Maximize
		rank = nsga_sort(objVals)

		# Assign ranks
		for i in range(len(self.pop)):
			self.pop[i].rank = rank[i]

	# def acculacy(self):
	# 	ind = sorted(self.pop, key=lambda i: i.rank)[0]
	# 	wVec = ind.wMat.flatten()
	# 	return self.task.getAcculacy(self.p, wVec, ind.aVec, False),\
	# 		self.task.getAcculacy(self.p, wVec, ind.aVec, True)

class Wann_max_nconn(src.wann.Wann):
	def probMoo(self):
		"""Rank population according to Pareto dominance.
		"""
		# Alternate second objective
		# Compile objectives
		maxFit  = np.asarray([ind.fitMax  for ind in self.pop])
		nConns  = np.asarray([ind.nConn   for ind in self.pop])
		nConns[nConns==0] = 1 # No conns is always pareto optimal (but boring)
		objVals = np.c_[maxFit, 1/nConns] # Maximize
		rank = nsga_sort(objVals)

		# Assign ranks
		for i in range(len(self.pop)):
			self.pop[i].rank = rank[i]

	# def acculacy(self):
	# 	ind = sorted(self.pop, key=lambda i: i.rank)[0]
	# 	wVec = ind.wMat.flatten()
	# 	return self.task.getAcculacy(self.p, wVec, ind.aVec, False),\
	# 		self.task.getAcculacy(self.p, wVec, ind.aVec, True)

class Wann_mean_max_max_nconn(src.wann.Wann):
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
			maxFit  = np.asarray([ind.fitMax  for ind in self.pop])
			nConns  = np.asarray([ind.nConn   for ind in self.pop])
			nConns[nConns==0] = 1 # No conns is always pareto optimal (but boring)
			objVals = np.c_[maxFit, 1/nConns] # Maximize
			rank = nsga_sort(objVals)

		# Assign ranks
		for i in range(len(self.pop)):
			self.pop[i].rank = rank[i]

	# def acculacy(self):
	# 	ind = sorted(self.pop, key=lambda i: i.rank)[0]
	# 	wVec = ind.wMat.flatten()
	# 	return self.task.getAcculacy(self.p, wVec, ind.aVec, False),\
	# 		self.task.getAcculacy(self.p, wVec, ind.aVec, True)


if __name__ == "__main__":
	print(hyp)
	wann = Wann_mean_nconn(hyp)
	print(wann.__class__.__name__)
	wann.train()
	print(hyp)

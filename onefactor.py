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
	"comment": "onefactor",
}

class Wann_mean(src.wann.Wann):
	def probMoo(self):
		self.pop.sort(key=lambda i: -i.fitness)
		# Assign ranks
		for i in range(len(self.pop)):
			self.pop[i].rank = i

class Wann_max(src.wann.Wann):
	def probMoo(self):
		self.pop.sort(key=lambda i: -i.fitMax)
		# Assign ranks
		for i in range(len(self.pop)):
			self.pop[i].rank = i

if __name__ == "__main__":
	print(hyp)
	wann = Wann_max(hyp)
	wann.train()
	print(hyp)

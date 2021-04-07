import src
import domain

alg_nVals = 6

hyp = {
	"task":"mnist256",
	"alg_wDist": "standard",
	"alg_nVals": alg_nVals,
	# "maxGen": 70 * 6 // alg_nVals,
	"maxGen": 4096 * 6 // alg_nVals,
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
	"ann_actRange": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

if __name__ == "__main__":
	print(hyp)
	wann = src.wann.Wann(hyp)
	wann.train()
	print(hyp)

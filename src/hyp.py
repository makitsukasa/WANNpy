# -- Hyperparameter plumbing --------------------------------------------- -- #
def loadHyp(pFileName, printHyp=False):
	"""Loads hyperparameters from .json file
	Args:
		pFileName - (string) - file name of hyperparameter file
		printHyp  - (bool)   - print contents of hyperparameter file to terminal?

	Note: see p/hypkey.txt for detailed hyperparameter description
	"""

	# Load Parameters from disk
	with open(pFileName) as data_file:
		hyp = json.load(data_file)

	# Task hyper parameters
	task = Task(games[hyp['task']],paramOnly=True)
	hyp['ann_nInput']   = task.nInput
	hyp['ann_nOutput']  = task.nOutput
	hyp['ann_initAct']  = task.activations[0]
	hyp['ann_actRange'] = task.actRange

	if 'alg_act' in hyp:
		hyp['ann_actRange'] = np.full_like(task.actRange,hyp['alg_act'])

	if printHyp is True:
		print(json.dumps(hyp, indent=4, sort_keys=True))
	return hyp

def updateHyp(hyp,pFileName=None):
	"""Overwrites default hyperparameters with those from second .json file
	"""
	print('\t*** Running with hyperparameters: ', pFileName, '\t***')
	''' Overwrites selected parameters those from file '''
	if pFileName != None:
		with open(pFileName) as data_file:
			update = json.load(data_file)

		hyp.update(update)

		# Task hyper parameters
		task = Task(games[hyp['task']],paramOnly=True)
		hyp['ann_nInput']   = task.nInput
		hyp['ann_nOutput']  = task.nOutput
		hyp['ann_initAct']  = task.activations[0]
		hyp['ann_actRange'] = task.actRange

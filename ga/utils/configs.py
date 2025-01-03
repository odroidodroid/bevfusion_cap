from datetime import datetime

current_time = datetime.now().strftime("%y%m%d-%H%M%S")
PROJECT_DIR = f"./ga/results/{current_time}_search"
CONDA_ENV = None
NUM_GPUS = 2
##### Configs
NGEN = 50
MU = 10     # population size
ELITE = int(MU*0.2) # elite size
LAMBDA = MU - ELITE # offspring size
CXPB = 0.7  # crossover probability
MUTPB = 0.3 # mutation probability

MINI_TRAIN_PKL = 'mini_infos_train.pkl'
MINI_VALID_PKL = 'mini_infos_val.pkl'

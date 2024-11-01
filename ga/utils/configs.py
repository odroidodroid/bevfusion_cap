from datetime import datetime

current_time = datetime.now().strftime("%y%m%d-%H%M%S")
PROJECT_DIR = f"./ga/results/{current_time}_search"
CONDA_ENV = 'bevfusion'
NUM_GPUS = 2
##### Configs
NGEN = 10
MU = 50     # population size
LAMBDA = 100
CXPB = 0.7  # crossover probability
MUTPB = 0.2 # mutation probability

MINI_TRAIN_PKL = 'mini_infos_train.pkl'
MINI_VALID_PKL = 'mini_infos_val.pkl'

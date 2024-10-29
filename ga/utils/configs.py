from datetime import datetime

current_time = datetime.now().strftime("%y%m%d-%H%M%S")
PROJECT_DIR = f"./ga/results/{current_time}_search"

##### Configs
NGEN = 50
MU = 30     # population size
LAMBDA = 100
CXPB = 0.7  # crossover probability
MUTPB = 0.2 # mutation probability
RESULT_FILE_PATH = f"{PROJECT_DIR}/ga_results.json"


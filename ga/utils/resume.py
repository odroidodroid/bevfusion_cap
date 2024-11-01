import os
import pickle
import json
from .configs import PROJECT_DIR
from .log import logger


if not os.path.exists(PROJECT_DIR):
    os.makedirs(PROJECT_DIR, exist_ok=True)


def save_checkpoint(pip, hall, logbook, gen):
    filename = os.path.join(PROJECT_DIR, f"checkpoint_{gen}.pkl")
    with open(filename, "wb") as f:
        pickle.dump((pip, hall, logbook, gen), f)
    logger.info(f"Checkpoint saved to {filename}")


def load_checkpoint(run_dir: str, gen: int = None):
    if gen is None:
        files = os.listdir(run_dir)
        files = [f for f in files if f.startswith("checkpoint_")]
        gen = max([int(f.split("_")[-1].split(".")[0]) for f in files])
    filename = os.path.join(run_dir, f"checkpoint_{gen}.pkl")
    with open(filename, "rb") as f:
        pip, hall, logbook, gen = pickle.load(f)
    return pip, hall, logbook, gen

def save_results(file_path : str, new_result : dict) :
    try :
        with open(file_path, 'r') as f :
            results = json.load(f)
    except FileNotFoundError :
        results = []
    results.append(new_result)
    
    with open(file_path,'w') as f :
        f.write('[')
        for idx, res in enumerate(results) :
            json.dump(res, f)
            if idx != (len(results)-1) :
                f.write(',\n')
            else :
                f.write(']')
    logger.info(f"evalKnapsck results saved to {file_path}")

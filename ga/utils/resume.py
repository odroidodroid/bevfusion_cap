import json
import os
import pickle

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


def save_results(file_path: str, new_result: dict):
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    results.append(new_result)

    with open(file_path, 'w') as f:
        json.dump(res, f, indent=4)
    logger.info(f"evalKnapsck results saved to {file_path}")


def load_cache(file_path : str) -> dict:
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cache from {file_path}. {e}")
        results = {}
    return results


def save_cache(file_path : str, cache : dict):
    with open(file_path, 'w') as f:
        json.dump(cache, f, indent=4)
    logger.info(f"Cache saved to {file_path}")

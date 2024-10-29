import json
import os

from .log import logger, write_stds
from .resume import load_checkpoint, save_checkpoint, save_results


def get_map(run_dir):
    json_file = None
    for file in os.listdir(run_dir):
        if file.endswith(".json"):
            json_file = os.path.join(run_dir, file)
            break

    if not json_file:
        logger.warning("No JSON file found in the run directory.")
        return 0.0

    with open(json_file, 'r') as f:
        lines = f.readlines()
        if not lines:
            logger.warning("JSON file is empty.")
            return 0.0

    for line in reversed(lines):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"{line} is not a JSON.")
            continue

        if data.get("mode") == "val" and data.get("epoch") == 6:
            return data.get("object/map") * 100 # mAP in percentage

    logger.warning("No matching mAP found in the JSON file.")
    return 0.0


def deep_update(original: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and key in original:
            deep_update(original[key], value)
        else:
            original[key] = value
    return original

import json
import os
import sqlite3

import pandas as pd

from .log import logger, write_stds
from .resume import (load_cache, load_checkpoint, save_cache, save_checkpoint,
                     save_results)


def get_map(run_dir):
    """Get the mAP from the JSON file."""
    # There is not only one json file in the run directory.
    # So, we need to find the right one.
    logfile = [file for file in os.listdir(run_dir) if file.endswith(".log")]
    if not logfile:
        logger.warning(f"No log file found in the run directory({run_dir}).")
        return 0.0

    json_file = os.path.join(run_dir, f"{logfile[0]}.json")
    if not os.path.exists(json_file):
        logger.warning(f"No JSON file found in the run directory({run_dir}).")
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

    logger.warning(f"No matching mAP found in the JSON file({json_file}).")
    return 0.0


def get_latency(run_dir):
    """
    Get the total latency from the NVTX_EVENTS table in the sqlite file.
    It must be called after 'nsys profile' command.
    """
    sqlite_file = os.path.join(run_dir, "report1.sqlite")
    
    if not os.path.exists(sqlite_file):
        logger.warning(f"No sqlite file found in the run directory({run_dir}).")
        return 1000.0
    
    latency = 1000.0
    try:
        with sqlite3.connect(sqlite_file) as conn:
            nvtx_df = pd.read_sql_query("SELECT start, end, textId FROM NVTX_EVENTS", conn)
            stringID_df = pd.read_sql_query("SELECT * FROM StringIds", conn)
        
        nvtx_df['duration'] = ((nvtx_df['end'] - nvtx_df['start']) / 1000000) # (ns -> ms)
        nvtx_df = pd.merge(nvtx_df, stringID_df, left_on='textId', right_on='id', how='left')
        nvtx_df = nvtx_df.drop(['textId', 'id', 'start', 'end'], axis=1)
        nvtx_df = nvtx_df.groupby(['value']).mean().reset_index()
        latency = nvtx_df['duration'].sum()
        
        # if visualize:
        #     title = model_name + ' profiling result(%)'
        #     nvtx_df.plot.pie(y='duration', figsize=(12, 5), title=title, autopct='%1.1f%%', xlabel='', ylabel='', legend=False)
        
        result: dict = nvtx_df[['value', 'duration']].set_index('value').to_dict()['duration']
        result['total'] = latency

        with open(os.path.join(run_dir, 'latency.json'), 'w') as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to get latency from {sqlite_file}. {e}")
    finally:
        return latency


def deep_update(original: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and key in original:
            deep_update(original[key], value)
        else:
            original[key] = value
    return original

import logging
import os
import select
from datetime import datetime

from .configs import PROJECT_DIR


if not os.path.exists(PROJECT_DIR):
    os.makedirs(PROJECT_DIR, exist_ok=True)

# Create a logger
current_time = datetime.now().strftime("%y%m%d-%H%M%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler(os.path.join(PROJECT_DIR, f"{current_time}_ga.log"), mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Stream handler (stdout)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def write_stds(process, enableStdout: bool = False, enableStderr: bool = False):
    with process.stdout as stdout, process.stderr as stderr:
        while True:
            reads = [stdout.fileno(), stderr.fileno()]
            ret = select.select(reads, [], [])
            for fd in ret[0]:
                if fd == stdout.fileno():
                    line = stdout.readline()
                    if line:
                        output = line.decode('utf-8').strip()
                        (logger.info if enableStdout else print)(output)
                if fd == stderr.fileno():
                    line = stderr.readline()
                    if line:
                        output = line.decode('utf-8').strip()
                        (logger.error if enableStderr else print)(output)
            if process.poll() is not None:
                break

# Test the logger
if __name__ == "__main__":
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

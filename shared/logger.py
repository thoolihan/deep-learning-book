import logging
import os
import inspect
from datetime import datetime


_start_time = None
_setup = False
_name = os.path.basename(inspect.stack()[-1][1]).replace(".py", "")
log_dir = "{}/../logs".format(os.path.dirname(__file__))

def get_filename():
    return _name

def get_curr_time():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

def get_start_time():
    return _start_time if _start_time else get_curr_time()

_start_time = get_start_time()

# Import and use this
def get_logger():
    if not(_setup):
        setup_logger()
    return logging.getLogger(_name)

# Never expose this
# Create a custom logger, because ai gym environment seems to hijack default logger
def setup_logger(level = logging.DEBUG):
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    cli_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    log = logging.getLogger(_name)

    cli = logging.StreamHandler()
    cli.setFormatter(cli_formatter)

    fl = logging.FileHandler("{}/{}-{}.txt".format(log_dir, get_start_time(), log.name))
    fl.setFormatter(file_formatter)

    log.handlers.clear()
    log.addHandler(cli)
    log.addHandler(fl)

    log.setLevel(level)
    log.propagate = False
    return log

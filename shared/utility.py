import os
from .logger import get_logger
from .plot_history import has_display

logger = get_logger()

def ensure_directory(path, logger = logger):
    if not os.path.exists(path):
        os.mkdir(path)
        if logger is not None:
            logger.info("Created directory: {}".format(path))
    else:
        if logger is not None:
            logger.info("Directory already exists: {}".format(path))


def open_plot(plot_file, logger = logger):
    if has_display():
        os_name = os.popen("uname -a").read()
        if "darwin" in os_name.lower():
            os.system("open {}".format(plot_file))
            return plot_file
    logger.info("open_plot only works on OS X")
import os
import platform
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
        os_name = platform.system().lower()
        if "darwin" in os_name:
            os.system("open {}".format(plot_file))
        elif "linux" in os_name:
            os.system("xdg-open {}".format(plot_file))
        elif "windows" in os_name:
            os.system("start {}".format(plot_file))
        else:
            logger.info("unrecognized platform.system()")
    else:
        logger.info("no display to open plot with")
    return plot_file
import os

def ensure_directory(path, logger = None):
    if not os.path.exists(path):
        os.mkdir(path)
        if logger is not None:
            logger.info("Created directory: {}".format(path))
    else:
        if logger is not None:
            logger.info("Directory already exists: {}".format(path))


def open_plot(plot_file):
    if os.name == "posix":
        os_name = os.popen("uname -a").read()
        if "darwin" in os_name.lower():
            os.system("open {}".format(plot_file))
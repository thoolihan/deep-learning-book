import os
import platform
from .logger import get_logger, get_start_time, get_filename
from .plot_history import has_display
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

TENSORBOARD_DIR=os.path.join('/tmp', 'tensorboard')
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
            os.system("gio open {}".format(plot_file))
        elif "windows" in os_name:
            os.system("start {}".format(plot_file))
        else:
            logger.info("unrecognized platform.system()")
    else:
        logger.info("no display to open plot with")
    return plot_file

def get_tensorboard_directory(project_name, start_time=get_start_time(), fname=get_filename(), tensorboard_dir=TENSORBOARD_DIR):
    return os.path.join(tensorboard_dir, "{}-{}-{}".format(project_name, fname, start_time))

def get_model_file(output_dir, fname=get_filename(), ts=get_start_time()):
    return os.path.join(output_dir, "model-{}-{}.h5".format(fname, ts))

def limit_gpu_memory(frac=0.75):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = frac
    set_session(tf.Session(config=config))

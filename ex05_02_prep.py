import os, shutil
from shared.logger import get_logger
import numpy as np

logger = get_logger()

project = "dogs_cats"

base_dir = os.path.join("data", project)
original_data_dir = os.path.join(base_dir, "original", "train")

def create_if_missing(path):
    if not os.path.exists(path):
        os.mkdir(path)
        logger.info("created directory: {}".format(path))
    else:
        logger.info("directory already exists: {}".format(path))

splits = {
    "train": range(1000),
    "validation": range(1000, 1500),
    "test": range(1500, 2000)
}
img_types = ['cat', 'dog']

for split, split_range in splits.items():
    dir_path = os.path.join(base_dir, split)
    create_if_missing(dir_path)
    for itype in img_types:
        type_path = os.path.join(dir_path, itype)
        create_if_missing(type_path)

splits = {
    "train": range(1000),
    "validation": range(1000, 1500),
    "test": range(1500, 2000)
}

for itype in img_types:
    for split, split_range in splits.items():
        fnames = ['{}.{}.jpg'.format(itype, i) for i in split_range]
        for fname in fnames:
            src = os.path.join(original_data_dir, fname)
            dst = os.path.join(base_dir, split, itype)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
        logger.info("copied {itype}.{imin}.jpg - {itype}.{imax}.jpg to {dest}".format(
            itype = itype, imin = np.min(split_range), imax = np.max(split_range), dest = os.path.join(dir_path, split)
        ))

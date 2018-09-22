import os, shutil
from shared.logger import get_logger
from shared.utility import ensure_directory
import numpy as np

logger = get_logger()

project = "dogs_cats"

base_dir = os.path.join("data", project)
ensure_directory(base_dir, logger)
original_data_dir = os.path.join(base_dir, "original", "train")

splits = {
    "train": range(1000),
    "validation": range(1000, 1500),
    "test": range(1500, 2000)
}
img_types = ['cat', 'dog']

for split, split_range in splits.items():
    dir_path = os.path.join(base_dir, split)
    ensure_directory(dir_path, logger)
    for itype in img_types:
        type_path = os.path.join(dir_path, "{}s".format(itype))
        ensure_directory(type_path, logger)

splits = {
    "train": range(1000),
    "validation": range(1000, 1500),
    "test": range(1500, 2000)
}

for itype in img_types:
    for split, split_range in splits.items():
        missing = 0
        fnames = ['{}.{}.jpg'.format(itype, i) for i in split_range]
        for fname in fnames:
            src = os.path.join(original_data_dir, fname)
            dst = os.path.join(base_dir, split, "{}s".format(itype), fname)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
                missing += 1
                if missing % 50 == 0:
                    logger.info("created {} files".format(missing))
        logger.info("copied {itype}.{imin}.jpg - {itype}.{imax}.jpg to {dest}".format(
            itype = itype, imin = np.min(split_range), imax = np.max(split_range), dest = os.path.join(dir_path, split)
        ))

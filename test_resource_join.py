from shared.donors.transform import resources_total
from shared.logger import get_logger, get_start_time, get_filename, get_curr_time
import numpy as np
import pandas as pd
import os

# Constants and Config for index, features, and label
PROJECT_NAME="donors"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
INDEX_COL="id"
LABEL = 'project_is_approved'
CREATE_OUTPUT = True
FEATURES = ['teacher_number_of_previously_posted_projects']
OHE_FEATURES = ['school_state', 'project_subject_categories', 'teacher_prefix', 'project_grade_category']
ESSAY_COLS =  ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']
COLS = [INDEX_COL] + FEATURES + OHE_FEATURES + ESSAY_COLS
COLS_TR = COLS + [LABEL]
DTYPES = {
    'id': object,
    'teacher_number_of_previously_posted_projects': int,
    'school_state': object,
    'project_sub_categories': object,
    'teacher_id': object,
    'teacher_prefix': object,
    'project_grade_category': object,
    'project_essay_1': object,
    'project_essay_2': object,
    'project_essay_3': object,
    'project_essay_4': object
}
RES_TYPES = {
    'id': object,
    'quantity': int,
    'price': np.float32
}
DTYPES_TR = DTYPES.copy()
DTYPES_TR[LABEL] = np.float32

logger = get_logger()
logger.info("running on {}".format(os.name))

# logger.info("reading original train file")
# train_df = pd.read_csv("{}/train.zip".format(INPUT_DIR), usecols=COLS_TR, index_col=INDEX_COL, dtype=DTYPES)
# train_df = train_df.sample(frac=1)

resources_df = pd.read_csv("{}/resources.zip".format(INPUT_DIR), usecols = RES_TYPES.keys(), dtype=RES_TYPES)
print(resources_df.head())
sums_df = resources_total(resources_df)
logger.info("type of sums: {}".format(type(sums_df)))
logger.info("shape of sums: {}".format(sums_df.shape))
print(sums_df.head())

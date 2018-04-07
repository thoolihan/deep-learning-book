import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from shared.logger import get_logger, get_start_time, get_filename, get_curr_time
from shared.donors.transform import count_essays, ESSAY_COUNT

# Constants and Config for index, features, and label
PROJECT_NAME="donors"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
INDEX_COL="id"
LABEL = 'project_is_approved'
CREATE_OUTPUT = True
FEATURES = ['teacher_number_of_previously_posted_projects']
COLS = [INDEX_COL] + FEATURES + ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']
COLS_TR = COLS + [LABEL]
DTYPES = {
    'teacher_number_of_previously_posted_projects': 'int64'
}
DTYPES_TR = DTYPES.copy()
DTYPES_TR[LABEL] = np.float32

logger = get_logger()
logger.info("running on {}".format(os.name))

# load data
logger.info("loading data")
train_df = pd.read_csv("{}/train.zip".format(INPUT_DIR), usecols=COLS_TR, index_col = INDEX_COL, dtype=DTYPES)
test_df = pd.read_csv("{}/test.zip".format(INPUT_DIR), usecols=COLS, index_col = INDEX_COL, dtype=DTYPES_TR)
train_df = train_df.sample(frac=1)

# count essays
count_essays(train_df)
count_essays(test_df)
FEATURES += [ESSAY_COUNT]

# model
model = LogisticRegression()
model.fit(X=train_df[FEATURES].as_matrix(), y=train_df[LABEL])
logger.info(model)

# write output
if CREATE_OUTPUT:
    logger.info("saving test results")
    answers = model.predict_proba(test_df[FEATURES].as_matrix())
    test_df[LABEL] = answers[:, 1]
    fname = "{}/{}-{}.csv".format(OUTPUT_DIR, get_filename(), get_start_time())
    test_df.to_csv(fname, index=True, columns = [LABEL])
    logger.info("created {}".format(fname))
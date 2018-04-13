import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from shared.logger import get_logger, get_start_time, get_filename, get_curr_time
from shared.donors.transform import count_essays, ESSAY_COUNT, resources_total
from hashlib import md5
from pathlib import Path

# Constants and Config for index, features, and label
PROJECT_NAME="donors"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
INDEX_COL="id"
LABEL = 'project_is_approved'
CREATE_OUTPUT = False
CLEAR_CACHE=True
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

# load data
logger.info("loading data")

# test_df has to be read regardless since it's used to create submission file
logger.info("reading test file")
test_df = pd.read_csv("{}/test.zip".format(INPUT_DIR), usecols=COLS, index_col = INDEX_COL, dtype=DTYPES_TR)

fields = ":".join(COLS)
hash = md5(fields.encode('utf-8')).hexdigest()
logger.debug("columns hash is {}".format(hash))
cached_numpy_file = Path("{}/cached/{}.npz".format(INPUT_DIR, hash))

if cached_numpy_file.exists() and not(CLEAR_CACHE):
    logger.info("reading cached files from numpy zip")

    with cached_numpy_file.open('rb') as fh:
        cached_info = np.load(fh)
        X_train = cached_info['X_train']
        y_train = cached_info['y_train']
        X_test = cached_info['X_test']
else:
    if CLEAR_CACHE:
        logger.info("CLEAR_CACHE is True")
    if not cached_numpy_file.exists():
        logger.info("{} does not exist".format(cached_numpy_file))
    logger.info("reading original train file")
    train_df = pd.read_csv("{}/train.zip".format(INPUT_DIR), usecols=COLS_TR, index_col = INDEX_COL, dtype=DTYPES)
    train_df = train_df.sample(frac=1)

    # load resources and do group sum
    res_df = pd.read_csv("{}/resources.zip".format(INPUT_DIR), usecols=RES_TYPES.keys(), dtype=RES_TYPES)
    sums_df = resources_total(res_df)

    # create features
    logger.info("counting essays")
    train_df = count_essays(train_df)
    train_df = train_df.drop(columns = ESSAY_COLS)
    test_df = count_essays(test_df)
    test_df = test_df.drop(columns = ESSAY_COLS)

    # add resource cost
    logger.info("add in resource totals")
    train_df = train_df.merge(sums_df, how='left', left_index=True, right_index=True)
    test_df = test_df.merge(sums_df, how='left', left_index=True, right_index=True)

    # Scale features
    logger.info("scaling columns")
    ss = StandardScaler()
    cont_cols = ['teacher_number_of_previously_posted_projects', 'total']
    ss.fit(train_df[cont_cols])
    train_df[cont_cols] = ss.transform(train_df[cont_cols])
    test_df[cont_cols] = ss.transform(test_df[cont_cols])

    # Encode features
    logger.info("encoding dummy variables")
    train_df = pd.get_dummies(train_df, columns = OHE_FEATURES)
    test_df = pd.get_dummies(test_df, columns = OHE_FEATURES)
    test_df.reindex(columns = train_df.drop(columns = [LABEL]).columns.values, fill_value=0)

    # to matrix
    logger.info("dataframes to matrices")
    X_train = train_df.drop(columns = [LABEL]).as_matrix()
    y_train = train_df[LABEL].values
    X_test = test_df.as_matrix()

    logger.info("caching numpy files")
    with cached_numpy_file.open('wb') as fh:
        np.savez(fh, X_train = X_train, y_train =  y_train, X_test = X_test)

# model
model = XGBClassifier()

logger.debug("Training data has shapes of X: {} y: {}".format(X_train.shape, y_train.shape))
logger.debug("Test data has shapes of X: {}".format(X_test.shape))

# evaluate model
scores = cross_val_score(model, X=X_train, y=y_train, cv=3, scoring='roc_auc')
logger.info("ROC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# fit all
model.fit(X=X_train, y=y_train)
logger.info(model)

# write output
if CREATE_OUTPUT:
    logger.info("saving test results")
    answers = model.predict_proba(X_test)
    test_df[LABEL] = answers[:, 1]
    fname = "{}/{}-{}.csv".format(OUTPUT_DIR, get_filename(), get_start_time())
    test_df.to_csv(fname, index=True, columns = [LABEL])
    logger.info("created {}".format(fname))
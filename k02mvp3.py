import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
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
OHE_FEATURES = ['school_state', 'project_subject_categories', 'teacher_id', 'teacher_prefix', 'project_grade_category']
ESSAY_COLS =  ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']
COLS = [INDEX_COL] + FEATURES + OHE_FEATURES + ESSAY_COLS
COLS_TR = COLS + [LABEL]
DTYPES = {
    'id': str,
    'teacher_number_of_previously_posted_projects': int,
    'school_state': str,
    'project_sub_categories': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'project_grade_category': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str
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

# create features
train_df = count_essays(train_df)
train_df = train_df.drop(columns = ESSAY_COLS)
test_df = count_essays(test_df)
test_df = test_df.drop(columns = ESSAY_COLS)

# Encode features
#  dict of encoders for later saving if need be
lencoders = {}
for col in OHE_FEATURES:
    # String to int
    lencoders[col] = LabelEncoder()
    lencoders[col].fit((train_df[col]).append(test_df[col]))
    train_df[col] = lencoders[col].transform(train_df[col])
    test_df[col] = lencoders[col].transform(test_df[col])

# to matrix
X_train = train_df.drop(columns = [LABEL]).as_matrix()
y_train = train_df[LABEL]
X_test = test_df.as_matrix()

# Categorical ints to bit field
col_mask = [col in OHE_FEATURES for col in train_df.drop(columns = [LABEL]).columns.values]
ohe = OneHotEncoder(sparse=False, categorical_features=col_mask, handle_unknown='ignore')
X_train = ohe.fit_transform(X_train)
X_test = ohe.fit_transform(X_test)

# model
model = LogisticRegression()

logger.debug("Training data has shapes of X: {} y: {}".format(X_train.shape, y_train.shape))
logger.debug("Test data has shapes of X: {}".format(X_test.shape))

# evaluate model
scores = cross_val_score(model, X=X_train, y=y_train, cv=10, scoring='roc_auc')
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
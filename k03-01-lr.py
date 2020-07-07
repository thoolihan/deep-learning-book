import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from shared.logger import get_logger, get_start_time, get_filename, get_curr_time
from shared.utility import ensure_directory
from sklearn.metrics import mean_squared_error

logger = get_logger()
logger.info("running on {}".format(os.name))

# Constants and Config for index, features, and label
PROJECT_NAME="santander"
INPUT_DIR="data/{}".format(PROJECT_NAME)
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
ensure_directory(OUTPUT_DIR, logger)
INDEX_COL="ID"
LABEL = 'target'
CREATE_OUTPUT = True
DTYPES = {
    INDEX_COL: 'str'
}

# load data
logger.info("loading data")
train_df = pd.read_csv("{}/train.csv".format(INPUT_DIR), index_col=INDEX_COL, dtype=DTYPES)
test_df = pd.read_csv("{}/test.csv".format(INPUT_DIR), index_col=INDEX_COL, dtype=DTYPES)

# Columns, etc
COLS = list(train_df)
FEATURES = [col for col in COLS if col not in [INDEX_COL, LABEL]]

# shuffle training data
train_df = train_df.sample(frac=1)
logger.info("Training data shape is {}".format(train_df.shape))

# model
model = LinearRegression()
model.fit(X=train_df[FEATURES].values, y=train_df[LABEL])
logger.info(model)

# write output
if CREATE_OUTPUT:
    logger.info("saving test results")
    answers = model.predict(test_df[FEATURES].values)
    test_df[LABEL] = [abs(x) for x in answers]
    fname = "{}/{}-{}.csv".format(OUTPUT_DIR, get_filename(), get_start_time())
    test_df.to_csv(fname, index=True, columns = [LABEL])
    logger.info("created {}".format(fname))

test_prediction = model.predict(test_df[FEATURES].values)
mse = mean_squared_error(test_df[LABEL], test_prediction)
print("RMSE of test: {}".format(mse))
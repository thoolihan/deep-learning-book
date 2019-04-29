import os
import numpy as np
import pandas as pd
from shared.logger import get_logger
import matplotlib.pyplot as plt
plt.style.use('ggplot')

logger = get_logger()

GLOVE_HOME = os.path.join(os.path.expanduser("~"), "workspace", "Embeddings", "glove")
DIMENSIONS = 100
EMBEDDINGS_FILE_NAME = "glove.6B.{}d.txt".format(DIMENSIONS)
EMBEDDINGS_PATH = os.path.join(GLOVE_HOME, EMBEDDINGS_FILE_NAME)
UPDATE_INDEX = 100000
PLOT_FILE = "w2v.png"
PROJECT_NAME="w2v"
OUTPUT_DIR="output/{}".format(PROJECT_NAME)
PLOT_PATH = os.path.join(OUTPUT_DIR, PLOT_FILE)

embeddings = {}
embeddings_file = open(EMBEDDINGS_PATH, mode="rt", encoding="utf-8")
logger.info("Reading embeddings from {}...".format(EMBEDDINGS_PATH))

for index, line in enumerate(embeddings_file):
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype = 'float32')
    embeddings[word] = vector
    if index % UPDATE_INDEX == 0:
        logger.info("Processed {} vectors".format(index))

embeddings_file.close()
logger.info("File closed. Read {} words".format(len(embeddings)))

seq = "hotels and apartments are sort of similar but the relation is complicated and simple at the same time".split()

df = pd.DataFrame(columns = ['x', 'y'])

for word in seq:
    vector = embeddings[word]
    df = df.append(pd.Series({'x': vector[0], 'y': vector[1]}, name = word))

logger.info("date is:")
logger.info(df)

plt.scatter(df.x, df.y)
for word, row in df.iterrows():
    plt.text(row.x, row.y, word)
logger.info("saving plot as: {}".format(PLOT_PATH))
plt.savefig(PLOT_PATH)      
plt.show()
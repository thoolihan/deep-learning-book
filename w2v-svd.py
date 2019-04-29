import os
import numpy as np
import pandas as pd
from shared.logger import get_logger
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
plt.style.use('ggplot')

logger = get_logger()

GLOVE_HOME = os.path.join(os.path.expanduser("~"), "workspace", "Embeddings", "glove")
DIMENSIONS = 100
EMBEDDINGS_FILE_NAME = "glove.6B.{}d.txt".format(DIMENSIONS)
EMBEDDINGS_PATH = os.path.join(GLOVE_HOME, EMBEDDINGS_FILE_NAME)
UPDATE_INDEX = 100000
PLOT_FILE = "w2v-svd.png"
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
        logger.info("Processed {:7,} vectors".format(index))

embeddings_file.close()
logger.info("File closed. Read {} words".format(len(embeddings)))

words = "hotels and apartments are sort of similar but the relation is complicated and simple at the same time".lower()
seq = words.split()

# make matrix of seq by embeddings
m = np.zeros((len(seq), DIMENSIONS))

for i, word in enumerate(seq):
    vector = embeddings[word]
    m[i,:] = vector

# decompose with NMF
decomp = TruncatedSVD(n_components=2)
components = decomp.fit_transform(m)

# create dataframe from that
df = pd.DataFrame(components, columns = ['x', 'y'], index = seq)

# show dataframe
logger.info("data is:")
logger.info(df)

# plot dataframe
plt.scatter(df.x, df.y)
for word, row in df.iterrows():
    plt.text(row.x, row.y, word)
logger.info("saving plot as: {}".format(PLOT_PATH))
plt.savefig(PLOT_PATH)  
plt.show()
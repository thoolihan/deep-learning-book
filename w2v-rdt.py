import os
import numpy as np
import pandas as pd
from shared.logger import get_logger
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import TweetTokenizer

plt.style.use('ggplot')
logger = get_logger()

GLOVE_HOME = os.path.join(os.path.expanduser("~"), "workspace", "Embeddings", "glove")
DIMENSIONS = 100
EMBEDDINGS_FILE_NAME = "glove.6B.{}d.txt".format(DIMENSIONS)
EMBEDDINGS_PATH = os.path.join(GLOVE_HOME, EMBEDDINGS_FILE_NAME)
UPDATE_INDEX = 100000

embeddings = dict()
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

twokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)

tweets = ["Kevin Stitt ran a great winning campaign against a very tough opponent in Oklahoma. Kevin is a very successful businessman who will be a fantastic Governor. He is strong on Crime & Borders, the 2nd Amendment, & loves our Military & Vets. He has my complete and total Endorsement!",
          "CNN is working frantically to find their “source.” Look hard because it doesn’t exist. Whatever was left of CNN’s credibility is now gone!",
          "Will be going to Evansville, Indiana, tonight for a big crowd rally with Mike Braun, a very successful businessman who is campaigning to be Indiana’s next U.S. Senator. He is strong on Crime & Borders, the 2nd Amendment, and loves our Military & Vets. Will be a big night!"
    ]

seqs = [(twokenizer.tokenize(tweet), i) for i, tweet in enumerate(tweets)]
seq = [word for seq, i in seqs for word in seq]
tweq = [i for seq, i in seqs for word in seq]

# make matrix of seq by embeddings
m = np.zeros((len(seq), DIMENSIONS))

for i, word in enumerate(seq):
    if word in embeddings:
        vector = embeddings[word]
        m[i,:] = vector

# decompose with NMF
decomp = TruncatedSVD(n_components=2)
components = decomp.fit_transform(m)

# create dataframe from that
df = pd.DataFrame(components, columns = ['x', 'y'], index = seq)
df['tweet'] = tweq

# show dataframe
logger.info("data is:")
logger.info(df)

# plot dataframe
plt.scatter(df.x, df.y, c = df.tweet)
for word, row in df.iterrows():
    plt.text(row.x, row.y, word)
plt.show()
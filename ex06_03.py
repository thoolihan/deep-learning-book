from shared.logger import get_logger
import numpy as np
from pprint import pprint

logger = get_logger()

samples = ['Mortal Kombat 11 released this week.', 'Street Fighter V has been out a while, something like 2016.']

dims = 1000
max_length = max([len(sample.split()) for sample in samples])

results = np.zeros(shape=(len(samples), max_length, dims))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dims
        results[i, j, index] = 1.
        
logger.info("results.shape: {}".format(results.shape))
pprint(results)
from tensorflow.keras.preprocessing.text import Tokenizer
from shared.logger import get_logger

logger = get_logger()

samples = ['Mortal Kombat 11 released this week.', 'this Street Fighter V has been out a while, something like 2016.']

tkn = Tokenizer(num_words=30)
tkn.fit_on_texts(samples)

sequences = tkn.texts_to_sequences(samples)
logger.info("sequences: {}".format(sequences))

one_hot_results = tkn.texts_to_matrix(samples, mode='binary')
logger.info("one_hot_results: {}".format(one_hot_results))

word_index = tkn.word_index
logger.info('Found {} unique tokens'.format(len(word_index)))

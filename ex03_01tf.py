from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import numpy as np
from shared.logger import get_logger
from shared.utility import limit_gpu_memory

limit_gpu_memory()
logger = get_logger()
EPOCHS = 5
BATCH_SIZE = 256
NUMWORDS = 5000
VALSIZE = 10000
def vectorize_sequence(sequences, dimension=NUMWORDS):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUMWORDS)

logger.info("Vectorizing data")
x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

y_train = np.asarray(train_labels, dtype='float32')
y_test = np.asarray(test_labels, dtype='float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(NUMWORDS,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

logger.debug("Compiling Model")
model.compile(optimizer=optimizers.RMSprop(lr=.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

logger.info("Fitting Model")
history = model.fit(x_train,
          y_train,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_split=0.3,
          shuffle=True)

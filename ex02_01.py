from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from shared.logger import get_logger
from shared.transform import flatten, scale

logger = get_logger()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess
train_images = scale(flatten(train_images))
test_images = scale(flatten(test_images))

# to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# nn
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer = 'rmsprop',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

# fit
model.fit(x=train_images, y=train_labels, epochs=5, batch_size=128)

# test
test_loss, test_acc = model.evaluate(x=test_images, y=test_labels)
logger.info("Test loss: {} Test accuracy: {}".format(test_loss, test_acc))

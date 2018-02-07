from keras.datasets import mnist
from shared.logger import get_logger
import matplotlib.pyplot as plt

logger = get_logger()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[4]

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# rotation of 2d data

data = np.array([[0, 0],
                 [1, 1],
                 [2, 2]], dtype='float32')

plt.plot(data[:, 0], data[:, 1], 'b:')
plt.plot(data[:, 0], data[:, 1], 'bs')

# 30 degree angle
theta = np.pi / 6

u = np.array([np.cos(theta), np.sin(theta)])
v = np.array([-np.sin(theta), np.cos(theta)])
rot = np.array([u, v])

data_t = np.dot(data, rot)

plt.plot(data_t[:, 0], data_t[:, 1], 'r:')
plt.plot(data_t[:, 0], data_t[:, 1], 'rs')
plt.show()


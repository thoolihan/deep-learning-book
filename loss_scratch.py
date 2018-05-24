import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

x = np.arange(0, 1, step = 0.05)
y = np.log(x)

plt.plot(x, y, 'b-')
#plt.show()

def likelihood(y_actual, y_pred_prob):
    return ((y_actual * y_pred_prob) + (1-y_actual) * (1 - y_pred_prob))

def log_loss(y_actual, y_pred_prob):
    return -1 * ((y_actual * np.log(y_pred_prob)) + (1-y_actual) * np.log(1 - y_pred_prob))

events = np.array([1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0])
pred = np.array([.7, .2, .05, .3, .4, .55, .49, .55, .65, .8, .9])

print("{} events and {} predictions".format(events.size, pred.size))
print("absolute misses: {}".format(np.abs(events - pred)))
print("mean absolute miss: {}\n".format(np.mean(np.abs(events - pred))))

li = likelihood(events, pred)
print("likelihood: {}".format(li))
print("likelihood total: {}\n".format(np.multiply.reduce(li)))

ll = log_loss(events, pred)
print("log loss: {}".format(ll))
print("log loss total: {}".format(np.sum(ll)))
print("mean log loss: {}".format(np.mean(ll)))
import numpy as np
from sklearn import metrics

__epsilon__ = 0.00000001

def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta = 1)

def fbeta_score(y_true, y_pred, beta):
    y_true = np.round(np.clip(y_true, 0, 1))
    y_pred = np.round(np.clip(y_pred, 0, 1))
    
    actual_true = np.sum(y_true, axis=0)
    predicted_true = np.sum(y_pred, axis=0)
    
    true_positive = np.sum(y_true * y_pred, axis=0)
    precision = true_positive / predicted_true
    recall = true_positive / actual_true
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    if denom.ndim >= 1:
        denom[denom == 0.] += __epsilon__
    else:
        if denom == 0:
            denom += __epsilon__
    return ((1 + beta_sq) * precision * recall) / denom

y_true = np.round(np.clip(np.random.rand(100, 3), 0, 1))
y_pred = np.round(np.clip(np.random.rand(100, 3), 0, 1))

print("custom f_score is  {}".format(f1_score(y_true, y_pred)))
print("sklearn f_score is {}".format(metrics.f1_score(y_true, y_pred, average = None)))

y_true1 = np.round(np.clip(np.random.rand(100), 0, 1))
y_pred1 = np.round(np.clip(np.random.rand(100), 0, 1))

print("custom f_score is  {}".format(f1_score(y_true1, y_pred1)))
print("sklearn f_score is {}".format(metrics.f1_score(y_true1, y_pred1)))
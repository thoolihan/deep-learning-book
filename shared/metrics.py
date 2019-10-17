import tensorflow.keras.backend as K

def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta = 1)

def fbeta_score(y_true, y_pred, beta):
    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))

    actual_true = K.sum(y_true, axis=0)
    predicted_true = K.sum(y_pred, axis=0)

    true_positive = K.sum(y_true * y_pred, axis=0)
    precision = true_positive / predicted_true
    recall = true_positive / actual_true
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall

    return K.mean(((1 + beta_sq) * precision * recall) / denom)

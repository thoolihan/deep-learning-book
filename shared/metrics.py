import keras.backend as K

def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    actual_true = K.sum(K.cast(K.greater(y_true, 0), K.floatx()))
    predicted_true = K.sum(K.cast(K.greater(y_pred, 0), K.floatx()))
    equal = K.cast(K.equal(y_true, y_pred), K.floatx())
    true_positive = K.sum(y_true * equal)
    precision = true_positive / predicted_true
    recall = true_positive / actual_true
    return 2 * (precision * recall) / (precision + recall)
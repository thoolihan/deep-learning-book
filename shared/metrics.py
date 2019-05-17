import keras.backend as K

def f1_score(y_true, y_pred):
    return f_score(y_true, y_pred, beta = 2)

def f_score(y_true, y_pred, beta = 2):
    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))
    
    actual_true = K.sum(y_true)
    predicted_true = K.sum(y_pred)
    
    true_positive = K.sum(y_true * y_pred)
    precision = true_positive / (predicted_true + K.epsilon())
    recall = true_positive / (actual_true + K.epsilon())
    return beta * (precision * recall) / (precision + recall + K.epsilon())
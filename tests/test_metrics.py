import unittest
from shared.metrics import f1_score
import keras.backend as K
import numpy as np
from sklearn.metrics import f1_score as sk_f1_score

class TestContinuous(unittest.TestCase):

    def test_fl_to_zero_ten(self):
        y_true = np.array([0., 0., 1., 1.], dtype=K.floatx())
        y_pred = np.array([0., 1., 1., 0.], dtype=K.floatx())
        y_true_k = K.variable(y_true)
        y_pred_k = K.variable(y_pred)

        expected = sk_f1_score(y_true, y_pred)
        actual = K.eval(f1_score(y_true_k, y_pred_k))

        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
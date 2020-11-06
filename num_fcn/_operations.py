import numpy as np
from ._ops_constants import RELU, SIGMOID, TAN, TRESHOLD_FUNC_HALF, LEAKY_RELU,\
    RELU_DERIV, SIGMOID_DERIV, TAN_DERIV, TRESHOLD_FUNC_HALF_DERIV, LEAKY_RELU_DERIV


class Ops:
    """
    Функции активации и их производные по числовому коду op
    """

    def __init__(self):
        self.alpha_leaky_relu = 1.7159
        self.alpha_sigmoid = 2
        self.alpha_tan = 1.7159
        self.beta_tan = 2/3

    def operations(self, op, x):
        """
        arg: op - числовой код (int)
             x  - точка графика
        return y:float     
        """
        y = np.zeros(x.shape[0])
        height = x.shape[0]
        if op == RELU:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = 0
                else:
                    y[row] = x[row][0]

            return np.array([y]).T

        elif op == RELU_DERIV:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = 0
            else:
                y[row] = 1
            return np.array([y]).T
        elif op == TRESHOLD_FUNC_HALF:
            for row in range(height):
                if (x[row][0] > 0.5):
                    y[row] = 1
                else:
                    y[row] = 0
            return np.array([y]).T
        elif op == TRESHOLD_FUNC_HALF_DERIV:
            return 1
        elif op == LEAKY_RELU:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = self.alpha_leaky_relu * x[row][0]
                else:
                    y[row] = x[row][0]
            return np.array([y]).T

        elif op == LEAKY_RELU_DERIV:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = self.alpha_leaky_relu
                else:
                    y[row] = 1
            return np.array([y]).T
        elif op == SIGMOID:
            y = 1 / (1 + np.exp(-self.alpha_sigmoid * x))
            return y
        elif op == SIGMOID_DERIV:
            return self.alpha_sigmoid * x * (1 - x)
        elif op == TAN:
            y = self.alpha_tan * np.tanh(self.beta_tan * x)
            return y
        elif op == TAN_DERIV:
            c=self.beta_tan/self.alpha_tan
            return c * (self.alpha_tan ** 2 - x ** 2) 
        else:
            print("Op or function does not support ", op)

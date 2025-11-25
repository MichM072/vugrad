from .core import Op, TensorNode

import numpy as np

class Relu(Op):
    """
    Op that applies a rectified linear unit activation function.
    """
    @staticmethod
    def forward(context, x):
        context['x'] = x

        return np.maximum(x, 0)

    @staticmethod
    def backward(context, go):
        x = context['x']
        gx = np.where(x > 0, 1, 0)

        return gx * go
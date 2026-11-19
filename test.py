import numpy as np

import h2ma


def ZDT1_1(x:np.ndarray)->np.ndarray:
    """ZDT1 test function"""
    return x[0]

def ZDT1_2(x:np.ndarray)->np.ndarray:
    """ZDT1 test function"""
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - (f1 / g) ** 0.5)
    return f2

a = h2ma.run([ZDT1_1, ZDT1_2], np.array([2.0, 11.0]), dim=2)
print(a)
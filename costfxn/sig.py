import numpy as np
def sigmoid(xt, beta):
    z = np.dot(xt, beta.T)
    return 1 / (1 + np.exp(-z))
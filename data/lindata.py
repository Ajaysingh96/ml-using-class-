import numpy as np
class Linear:
    def __init__(self):
        np.random.seed(2000)
        self.m = np.random.random(1)
        self.c = np.random.random(1)
        self.x = (np.random.normal(loc=0.0, scale=10.0, size=100))
        self.e = np.random.normal(loc=0, scale=1, size=100)
        self.data()
    def data(self):
        self.Y_error=self.m*self.x+self.c+self.e
        self.X=self.x



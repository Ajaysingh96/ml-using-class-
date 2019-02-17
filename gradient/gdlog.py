from costfxn.crossentropy import Logisticreg
from costfxn.sig import sigmoid
import numpy as np
class Gradient:
    def __init__(self,x,y,theta,num):
        self.X = x
        self.Y = y
        self.beta = theta
        self.n=num
        self.gdescent(self.X,self.Y, self.beta,self.n)



    def gdescent(self, xdata, label,betas, n, l=0.1):
        for i in range(n):
            betas = betas - ((l / xdata.shape[0]) * np.dot(xdata.T, sigmoid(xdata, betas) - label))
        return betas
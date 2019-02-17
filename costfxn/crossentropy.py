import numpy as np
from costfxn.sig import sigmoid
class Logisticreg:
    def __init__(self,x,y,theta):
        self.X=x
        self.Y=y
        self.beta=theta
        self.costfxn(self.X,self.Y,self.beta)


    def costfxn(self,X, y,theta):
        cc = -y * np.log(sigmoid(X,theta)) - (1 - y) * np.log(1 - sigmoid(X,theta))
        return np.mean(cc)

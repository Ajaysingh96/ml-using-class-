import numpy as np
class Linearreg:
    def __init__(self,x,y,slope,constant):
        self.X=x
        self.Y=y
        self.m= slope
        self.c=constant
        self.costfxn(self.X,self.Y,self.m,self.c)

    def costfxn(self,x,y1,m1,c1):
        cc=(y1 - (m1 * x + c1)) * (y1 - (m1 * x + c1))
        return np.sum(cc)
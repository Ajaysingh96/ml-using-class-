from costfxn.errorsq import Linearreg
import numpy as np

class Gradient:
    def __init__(self,x,y,slope,constant):
        self.X = x
        self.Y = y
        self.m = slope
        self.c = constant
        self.fxn(self.X, self.Y, self.c, self.m)

    def fxn(self,x,y1,c1,m1,ll=0.01):   #ll=learning rate
        obj=Linearreg(x,y1,m1,c1)
        i = 1
        l=[]
        mm=[]
        cc=[]
        while i < 3000:
            m1 = m1 - ll * (-1 / len(y1)) * np.sum((y1 - (m1 * x + c1)) * x)
            c1 = c1 - ll * (-1 / len(y1)) * np.sum((y1 - (m1 * x + c1)))
            cost=obj.costfxn(x,y1,m1,c1)
            l.append(cost)
            mm.append(m1)
            cc.append(c1)
            i += 1
        index=np.argmin(l)

        return mm[index],cc[index]

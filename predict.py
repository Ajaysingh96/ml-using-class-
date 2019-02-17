from Models import linear
import numpy as np
from Models import logistic
from costfxn.sig import sigmoid
import matplotlib.pyplot as plt


x=[1,0.050251,4.88509]                            #logistic
theta=logistic.grad()
prediction=sigmoid(x,theta)
np.round(prediction)
#----------------------------------------------------------------------------------

para=linear.grad()                          #linear
m=para[0]
c=para[1]
y_error=linear.label
x=linear.x            #put value of x and get predicted value
pred=m*x+c


plt.scatter(x,y_error)
plt.plot(x,pred)
plt.show()






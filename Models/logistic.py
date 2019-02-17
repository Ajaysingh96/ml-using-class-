from costfxn.crossentropy import Logisticreg
from data import logdata
import numpy as np
from gradient.gdlog import Gradient
from sklearn.model_selection import train_test_split

obj=logdata.Logistic_reg_data()
data1=obj.data    #without  ones
label=obj.Y
m1=label.shape
m=m1[0]
def addones(xx,m):
    oness = np.ones((m, 1))
    X1 = np.append(oness,xx, axis=1)
    return X1
data = addones(data1,m)
n=data.shape
initial_theta=np.zeros(n[1])
data_train,data_test,label_train,label_test=train_test_split(data,label,test_size=0.5,stratify=label)
def costfunction(initial_theta):
    obj_for_logisticregg = Logisticreg(data_train, label_train, initial_theta)
    cc=obj_for_logisticregg.costfxn(data_train, label_train, initial_theta)
    return cc

def grad():
    objforgd = Gradient(data_train, label_train, initial_theta, 2000)
    best_theta = objforgd.gdescent(data_train, label_train, initial_theta, 2000)
    return best_theta


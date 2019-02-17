from data.lindata import Linear
from costfxn.errorsq import Linearreg
from gradient.gdlin import Gradient
from sklearn.model_selection import train_test_split
obj=Linear()
m=obj.m
c=obj.c
x=obj.x
label=obj.Y_error
data_train,data_test,label_train,label_test=train_test_split(x,label,test_size=0.5)
def costfunction(m,c):
    obj_forlinear_cf = Linearreg(data_train, label_train, m, c)
    return obj_forlinear_cf.costfxn(data_train, label_train, m, c)
def grad():
    obj_for_gd = Gradient(data_train, label_train, m, c)
    p=obj_for_gd.fxn(data_train, label_train, m, c)
    return p




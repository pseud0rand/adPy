# Simple Audomatic Differentiation Class
# Forward Form
#
# valder stands for value, derivative
#
# Only special function sin(), cos(), and exp() are defined. Add more below as needed.
#
# The class valder has two properties, value and derivative, coresponding to the value and derivative.
# 
# Example:
# from adpy_forward_mode import *
#
# x = valder(3,4)
# y = valder(1,5)
# z = x*exp(y)
#
# print(z.value)
#   8.154845485377136
# print(z.derivative)
#   51.64735474072185
#

import numpy as np

class valder:

    def __init__(self,x,dx):
        self.value  = np.array(x)
        self.derivative = np.array(dx)

    def __mul__(self,vobj):
        return valder(self.value*vobj.value,self.value*vobj.derivative+self.derivative*vobj.value)

    def __div__(self,vobj):
        return valder(self.value/vobj.value,-self.value/vobj.derivative**2+1.0/vobj.value)

    def __add__(self,vobj):
        return valder(self.value+vobj.value,self.derivative+vobj.derivative)

    def __sub__(self,vobj):
        return valder(self.value-vobj.value,self.derivative-vobj.derivative)

# Special Functions
def sin(vobj):
    return valder(np.sin(vobj.value),vobj.derivative*np.cos(vobj.value))

def cos(vobj):
    return valder(np.cos(vobj.value),-vobj.derivative*np.sin(vobj.value))

def exp(vobj):
	return valder(np.exp(vobj.value),vobj.derivative*np.exp(vobj.value))
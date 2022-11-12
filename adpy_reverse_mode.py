# Simple Audomatic Differentiation Class
# Reverse form
#
# vallder stands for value,local-derivative
# - local means that it takes the derivative of the immediate dependent variable only, not going down the dependence tree
#
# vallder objects have the following properties:
# self.name  -- name of the object
# self.value -- value of the object, essentially obtained from evaluated name
# self. loc_ders -- values of the local derivatives of the object (dictionary)
#
# Heavy use of dictionaries is made to create object names. These are parsed to evaluate the corresponding expression.
#
# The most important function here is get_ders(), which is the back-propagation step, collecting all derivatives.
#
#
#
#Example:
#--------
#In [1]: from adpy_reverse_mode import *
#In [2]: x=vallder('x',3)
#In [3]: y=vallder('y',5)
#In [4]: z=vallder('z',8)
#In [5]: h=x*sin(x*y)
#In [6]: vars(h)
#Out[6]:
#{'name': 'x*sinx*y',
# 'value': 1.9508635204713505,
# 'loc_ders': {'x*sin(x*y)_sinx*y': 3, 'x*sin(x*y)_x': 0.6502878401571168}}
#
#Another example, which still contains some diagnostic print() statements:
#
#In [47]: x=vallder('x',3)
#In [48]: y=vallder('y',5)
#In [49]: z=vallder('z',8)
#
#In [50]: h=x*z*sin(x*y)
#{'name': 'x*z', 'value': 24, 'loc_ders': {'x*z_z': 3, 'x*z_x': 8}}
#{'name': 'x*y', 'value': 15, 'loc_ders': {'x*y_y': 3, 'x*y_x': 5}}
#{'name': 'sin(x*y)', 'value': 0.6502878401571168, 'loc_ders': {'sin(x*y)_x*y': -0.7596879128588213}}
#{'name': 'x*z*sin(x*y)', 'value': 15.606908163770804, 'loc_ders': {'x*z*sin(x*y)_sin(x*y)': 24, 'x*z*sin(x*y)_x*z': 0.6502878401571168}}
#
#In [51]: get_ders(h,x,['x','y','z'])
#x*z*sin(x*y)_sin(x*y), var=x, 2
#{'name': 'x*y', 'value': 15, 'loc_ders': {'x*y_y': 3, 'x*y_x': 5}}
#{'name': 'sin(x*y)', 'value': 0.6502878401571168, 'loc_ders': {'sin(x*y)_x*y': -0.7596879128588213}}
#{'name': 'x*z', 'value': 24, 'loc_ders': {'x*z_z': resdict3, 'x*z_x': 8}}
#sin(x*y)_x*y, var=x, 1
#{'name': 'x*y', 'value': 15, 'loc_ders': {'x*y_y': 3, 'x*y_x': 5}}
#x*y_y, var=x, 2
#x_x, var=x, 1
#x*z_z, var=x, 2
#x_x, var=x, 1
#Out[51]: -85.96024682180163
#
import numpy as np
from string import digits

# operators used for parsing expressions
ops = ['+','-','/','*',')','sin(','cos(','tan(','exp(','sinh(','cosh(','tanh(']

# dictionary containing results of the forward pass, and local derivatives
resdict = []

var_name_elements='a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,1,2,3,4,5,6,7,8,9,0'
var_name_elements=np.array(var_name_elements.split(','))


class vallder:

    def __init__(self,name,value):
        self.name=name
        self.value=value
        self.loc_ders={'%s_%s'%(name,name):1}
        pass

    def __mul__(self,vobj):
        global resdict
             
        res = vallder(self.name+'*'+vobj.name,self.value*vobj.value)
        res.loc_ders={res.name+'_'+vobj.name:self.value,res.name+'_'+self.name:vobj.value}        
        resdict.append(res) # dictinary containing results
        return res

    def __div__(self,vobj):
        global resdict
             
        res = vallder(self.name+'/'+vobj.name,self.value/vobj.value)
        res.loc_ders={res.name+'_'+vobj.name:-self.value/vobj.value**2,res.name+'_'+self.name:1/vobj.value}        
        resdict.append(res) # dictinary containing results
        return res

    def __add__(self,vobj):
        global resdict
        
        res = vallder('('+self.name+'+'+vobj.name+')',self.value+vobj.value)
        res.loc_ders={res.name+'_'+vobj.name:1.0,res.name+'_'+self.name:1.0}        
        resdict.append(res) # dictinary containing results
        return res       	

    def __sub__(self,vobj):
        global resdict
        
        res = vallder('('+self.name+'-'+vobj.name+')',self.value-vobj.value)
        res.loc_ders={res.name+'_'+vobj.name:-1.0,res.name+'_'+self.name:1.0}        
        resdict.append(res) # dictinary containing results
        return res       	

def getvars(expr):
    # finds variables in an expression, parsing done using operator array ops
    global ops
    
    for o in ops:
        expr = expr.replace(o,' ')
    expr = expr.strip()
    exprarr = expr.split()
    return np.unique(exprarr)

def get_ders(vobj,var,indep_vars):
	# Get partial derivative of vobj with respect to var
	# This function is recursive, calling itself
	# This function defines random variables using exec(), dynamically assigning values to variables, or vallder objects.
	# This is the way the results from individual leafs of the binary tree are collected.
	# The randomly named variables, or vallder objects, must be global for exec() to work. The randomly named vallder objects are accessed using globals()[v0_name]
	# and globals()[v1_name], where v0_name and v1_name are strings correspondign to the random names of the vallder objects.
    # Since the names of the vallder objects are random, the global space gets filled up with these, and it might be
	# necessary to clean it to save space eventually.
    #
	#
	# Potentially rename this function as collect_derivatives, or reverse_propagation or something that emphasized that this is
	# the back-propagation step.
	#
	# vobj = vallder object
	# var  = vallder object of the independent variable with respect to which to take the derivative
	# indep_var = array of strings corresponding to the names of the independent variables
	#
	# The result, res, is a "dot product" of the leafs of the binary tree, if the vobj has two elements in the loc_ders dictionary.
	# That is, in pseudo-code, res = [loc_ders[0], loc_ders[1]] (dot) [get_ders(v0),get_ders(v1)], where v0 and v1 are randomly named global vallder objects, 
	# defined by extracting the "sub" part of the vobj.loc_ders.
	#
	# The indep_vars list is passed to this function just to make sure to zero out the derivatives of all other independent variables than var.
	# The order of the if-statments is important, to first check whether the new v0_str or v1_str is equal to var.name, and if not, then also make sure
	# that it is not equal to other variables, with v0_str in indep_vars, and v1_str in indep_vars.
	#
	keys = list(vobj.loc_ders.keys())
	f0_str = keys[0].split('_')[0]
	v0_str = keys[0].split('_')[1]
	v0_name = ''.join(var_name_elements[np.random.randint(len(var_name_elements),size=10)])
	v0_name = v0_name.lstrip(digits)
	print('%s, var=%s, %i'%(keys[0],var.name,len(keys)))
	
	if len(keys)==1:			
		if v0_str == var.name:			
			res = list(vobj.loc_ders.values())[0]*vobj.loc_ders[keys[0]]
		elif v0_str in indep_vars:
			res = 0
		elif v0_str == f0_str:
			res = 1
		else:
			exec('global %s; %s = %s'%(v0_name,v0_name,v0_str))			
			res = list(vobj.loc_ders.values())[0]*get_ders(globals()[v0_name],var,indep_vars)		
	else:
		v1_str = keys[1].split('_')[1]	
		v1_name = ''.join(var_name_elements[np.random.randint(len(var_name_elements),size=10)])
		v1_name = v1_name.lstrip(digits)
		if v0_str == var.name and v1_str == var.name:			
			res = np.dot(list(vobj.loc_ders.values()),[vobj.loc_ders[keys[0]],vobj.loc_ders[keys[1]]])		
		elif v0_str == var.name:
			exec('global %s; %s = %s'%(v1_name,v1_name,v1_str))	
			res = np.dot(list(vobj.loc_ders.values()),[vobj.loc_ders[keys[0]],get_ders(globals()[v1_name],var,indep_vars)])
		elif v0_str in indep_vars:
			exec('global %s; %s = %s'%(v1_name,v1_name,v1_str))	
			res = np.dot(list(vobj.loc_ders.values()),[0,get_ders(globals()[v1_name],var,indep_vars)])
		elif v1_str == var.name:
			exec('global %s; %s = %s'%(v0_name,v0_name,v0_str))
			res = np.dot(list(vobj.loc_ders.values()),[get_ders(globals()[v0_name],var,indep_vars),vobj.loc_ders[keys[1]]])
		elif v1_str in indep_vars:
			exec('global %s; %s = %s'%(v0_name,v0_name,v0_str))
			res = np.dot(list(vobj.loc_ders.values()),[get_ders(globals()[v0_name],var,indep_vars),0])
		else:
			exec('global %s; %s = %s'%(v0_name,v0_name,v0_str))
			exec('global %s; %s = %s'%(v1_name,v1_name,v1_str))		
			res = np.dot(list(vobj.loc_ders.values()),[get_ders(globals()[v0_name],var,indep_vars),get_ders(globals()[v1_name],var,indep_vars)])
	return res
	
def sin(vobj):    
    global resdict
    
    res = vallder('sin('+vobj.name+')',np.sin(vobj.value))
    res.loc_ders = {res.name+'_'+vobj.name:np.cos(vobj.value)}
    print(vars(res))
    resdict.append(res)
    return res

def cos(vobj):    
    c = vallder('cos('+vobj.name+')',np.cos(vobj.value))
    c.loc_ders = {c.name+'_'+vobj.name:-np.sin(vobj.value)}
    return c

def tan(vobj):
	c = vallder('tan('+vobj.name+')',np.tan(vobj.value))
	c.loc_ders = {c.name+'_'+vobj.name:1.0/np.cos(vobj.value)**2}
	return c
	
def exp(vobj):
	c = vallder('exp('+vobj.name+')',np.exp(vobj.value))
	c.loc_ders = {c.name+'_'+vobj.name:np.exp(vobj.value)}
	return c
	
def tanh(vobj):
	c = vallder('tanh('+vobj.name+')',np.tanh(vobj.value))
	c.loc_ders = {c.name+'_'+vobj.name:1.0 - np.tanh(vobj.value)**2}
	return c
	
def cosh(vobj):
	c = vallder('cosh('+vobj.name+')',np.cosh(vobj.value))
	c.loc_ders = {c.name+'_'+vobj.name:np.sinh(vobj.value)}
	return c
		
def sinh(vobj):
	c = vallder('sinh('+vobj.name+')',np.sinh(vobj.value))
	c.loc_ders = {c.name+'_'+vobj.name:np.cosh(vobj.value)}
	return c
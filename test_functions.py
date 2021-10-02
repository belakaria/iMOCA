import math
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy


    
def cmfCurrin(x1,d,f1):
    x=deepcopy(x1)
    f=deepcopy(f1)
    f=(f+1)/20
    if x[1]==0:
        x[1]=1e-100
    return -1*float(((1 -0.1*(1-f)-math.exp(-0.5*(1/x[1]))) * ((2300*pow(x[0],3) + 1900*x[0]*x[0] + 2092*x[0] + 60)/(100*pow(x[0],3) + 500*x[0]*x[0] + 4*x[0] + 20))))

def cmfbranin(x1,d,f1):
    x=deepcopy(x1)
    f=deepcopy(f1)
    f=(f+1)/20
    x[0]= 15* x[0]-5
    x[1]=15*x[1]
    return -1*float(np.square(x[1] - (5.1/(4*np.square(math.pi)) - 0.01*(1-f))*np.square(x[0]) + (5/math.pi - 0.1*(1-f))*x[0]- 6) + 10*(1-(1./(8*math.pi) + 0.05*(1-f)))*np.cos(x[0]) + 10)

##############################################################
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
plt.rcParams.update({'font.size': 23})
import pandas as pd
import scipy.optimize #for confidence intervals
import matplotlib
from lmfit import minimize, Parameters
import seaborn as sns


def ExpDecayfit(x, y, err=None, supposeSqrt=False, start_at_from_top=0.95, exp_num:int=3, x0_vary:bool=True,xmax:float=0):
    
    cut=start_at_from_top
    x_0Guess = x[np.argmax(y)]
    OffsetGuess=np.mean(y[-20:-3])

    if err==None:
        err=np.ones(len(x))

    if supposeSqrt:
        err=np.sqrt(y)
        np.where(y==0, 1, y)

    y        = y.astype(np.float)
    x        = x.astype(np.float)
    AmpGuess = np.max(y) #cioè 1
    arg_max  = np.argmax(y)
    y        = y[arg_max:]
    x        = x[arg_max:]
    err      = err[arg_max:]
    x        = x[y<np.max(y)*cut]
    err      = err[y<np.max(y)*cut]
    y        = y[y<np.max(y)*cut]
    
    if xmax!=0:
        xmax+= x_0Guess
        y    = y[x<xmax]
        err  = err[x<xmax]
        x    = x[x<xmax]
    
    #tauGuess = x[np.where(y > AmpGuess/np.e)[0][-1]] - x_0Guess
    tauGuess=1
    #print('tauGuess=',tauGuess)
    #OffsetGuess = 135#y.min()#np.mean(y[-20:])
    
    params = Parameters()
    params.add('Amp_1',   value = AmpGuess/3,  min=0)
    if (exp_num>1):
        params.add('Amp_2',   value = AmpGuess/3,  min=0)
    else:
        params.add('Amp_2',   value = 0,  min=0, vary=False)
    
    if (exp_num>2):
        params.add('Amp_3',   value = AmpGuess/3,  min=0)
    else:
        params.add('Amp_3',   value = 0,  min=0, vary=False)
    
    #params.add('Amp_3',   value = AmpGuess/3,  min=0)
    params.add('x_0', value = x_0Guess, vary=x0_vary)
    params.add('tau_1', value = tauGuess/2, min=0)
    if (exp_num>1):
        params.add('tau_2', value = tauGuess, min=0)
    else:
        params.add('tau_2', value = tauGuess/2, min=0, vary=False)
#    params.add('tau_2', value = tauGuess, min=0)
    if (exp_num>1):
        params.add('tau_3', value = tauGuess*3, min=0)
    else:
        params.add('tau_3', value = tauGuess*3, min=0, vary=False)
    params.add('Offset', value = OffsetGuess, min=10)

    def fcn2min(par, x, y):

        x_0      = par['x_0'].value
        Amp_1    = par['Amp_1'].value
        Amp_2    = par['Amp_2'].value
        Amp_3    = par['Amp_3'].value
        tau_1    = par['tau_1'].value
        tau_2    = par['tau_2'].value
        tau_3    = par['tau_3'].value
        Offset   = par['Offset'].value
        model    = Amp_1*np.exp(-(x-x_0)/tau_1) + Amp_2*np.exp(-(x-x_0)/tau_2) + Amp_3*np.exp(-(x-x_0)/tau_3) + Offset
        resids   = model - y
        weighted = np.sqrt(resids ** 2 / err ** 2)

        return weighted


# do fit, here with leastsq model
    result = minimize(fcn2min, params, args = (x, y))
    # calculate final result
    par     = result.params
    x_0     = par['x_0'].value
    Amp_1   = par['Amp_1'].value
    Amp_2   = par['Amp_2'].value
    Amp_3   = par['Amp_3'].value
    tau_1   = par['tau_1'].value
    tau_2   = par['tau_2'].value
    tau_3   = par['tau_3'].value
    Offset  = par['Offset'].value
    Fit     = Amp_1*np.exp(-(x-x_0)/tau_1) + Amp_2*np.exp(-(x-x_0)/tau_2) + Amp_3*np.exp(-(x-x_0)/tau_3) + Offset

   # Fit = y + np.sqrt(result.residual**2 * err**2)
    Guesses = [AmpGuess, x_0Guess, tauGuess, OffsetGuess]
#    p = [result.params['Amp_1'].value, result.params['x_0'].value, result.params['tau_1'].value, result.params['tau_2'].value]
#    Std = [result.params['Amp_1'].stderr, result.params['x_0'].stderr, result.params['tau_1'].stderr, result.params['tau_2'].stderr]
    p=result.params
    return x, Fit, p#, Std, Guesses


def objective(limit, target, H):
    w = np.where(H>limit)
    count = H[w]
    return count.sum() - target

class fotone:
    #    Dtime :int=0
    #    TimeTag : int=0
        def __init__(self, TimeTag: int, Dtime: int):
            self.Dtime=Dtime
            self.TimeTag=TimeTag
    
class hist_bin:
    
    def __init__(self, bin_size: float):
        self.vall=[] #vector with all the photon in the bin
        if (bin_size<=0):
            raise Exception("bin_size cannot be zero or negative. The value of bin_size was {}".format(bin_size))
        self.bin_size=bin_size
        
    def append(self, fotone: fotone):
        self.vall.append(fotone)
        
    def getintensity(self, bin_size):
        self.intensity=(len(self.vall)/bin_size)
        return self.intensity
    
    def mean_time(self, Resolution, tmin:float=0, tmax:float=0):
        if tmax==0:
            filt=False
        else:
            filt=True
        Dtimes=np.array([el.Dtime for el in self.vall])*Resolution
        if filt:
            Filt_Dtimes=Dtimes[(Dtimes<=tmax)&(Dtimes>=tmin)]
        else:
            Filt_Dtimes=Dtimes
        self.meantime=Filt_Dtimes[~np.isnan(Filt_Dtimes)].mean()
        return self.meantime

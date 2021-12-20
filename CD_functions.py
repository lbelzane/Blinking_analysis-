import numpy as np 
import pandas as pd

'''
Fonction needed for the CD python file.

'''
def histc(X, bins):                                    
    '''
    This function return a list with the number of occurences in each bin and 
    the indices of the bins to which each value in input array belongs.
    '''
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(len(bins))
    for i in map_to_bins:
        r[i-1] += 1
    return [r, map_to_bins]


def ON(ONLevel, n, tauBlinking, Timetrace, df):
    '''
    Return a list with the cumulative desity of probability for ON occurences.
    '''
    
    IndexON = df[Timetrace > ONLevel].index.tolist()

    if len(IndexON)>2:
        Nph_MeanON = np.mean(Timetrace[IndexON])
        diffIndexON = np.diff(IndexON)
        IndexdiffIndexON = np.where(diffIndexON > 1)[0]

        if len(IndexdiffIndexON) == 0:
            return []
            
        else:
            tauON = np.diff(IndexdiffIndexON)*n 
            tauON = tauON.tolist()
            tauON.insert(0,IndexdiffIndexON[0]*n)
            tauON[0]+=n 
            tauON.append((len(diffIndexON)-IndexdiffIndexON[-1]+1)*n)

            BlinkingON    = histc(tauON, tauBlinking)[0]
            BlinkingONCUM = np.cumsum(BlinkingON[::-1])
            BlinkingONCUM = (np.asarray(BlinkingONCUM[::-1])/np.max(BlinkingONCUM))
            return  BlinkingONCUM
    else: 
        return []

    


def OFF(OFFLevel, n, tauBlinking, Timetrace, df):
    '''
    Return a list with the cumulative desity of probability for ON occurences.
    '''

    IndexOFF = df[Timetrace < OFFLevel].index.tolist()

    if len(IndexOFF)>2:
        Nph_MeanOFF       = np.mean(Timetrace[IndexOFF])
        diffIndexOFF      = np.diff(IndexOFF)
        IndexdiffIndexOFF = np.where(diffIndexOFF > 1)[0]

        if len(IndexdiffIndexOFF) == 0:
            return []

        else: 
            tauOFF = np.diff(IndexdiffIndexOFF)*n 
            tauOFF = tauOFF.tolist()
            tauOFF.insert(0,IndexdiffIndexOFF[0]*n)
            tauOFF[0]+=n 
            tauOFF.append((len(diffIndexOFF)-IndexdiffIndexOFF[-1]+1)*n)

            BlinkingOFF    = histc(tauOFF, tauBlinking)[0]
            BlinkingOFFCUM = np.cumsum(BlinkingOFF[::-1])
            BlinkingOFFCUM = (np.asarray(BlinkingOFFCUM[::-1])/np.max(BlinkingOFFCUM))
            return  BlinkingOFFCUM
    else:
        return []



def model_1(x,a,b):
    '''
    Model Power law
    '''
    return np.exp(b)*(x**a)

def model_2(x,tau_0, mu, Theta, tau_c, C):
    '''
    Model Power law + exponential
    ''' 
    return C*(((tau_0/x)**mu - (tau_0/Theta)**mu)/(1-(tau_0/Theta)**mu))*np.exp(-x/tau_c) 

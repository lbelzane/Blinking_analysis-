import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
import os
from data_load import *


def counts(Timetrace):
    '''
    This function takes a Timetrace in argument and return the list of intensity and a list of occurences 
    for each intensity.
    '''
    timetrace_sorted = sorted(Timetrace)
    occurences = []
    intensity  = []
    for k in range(len(timetrace_sorted)):
        if timetrace_sorted[k] != timetrace_sorted[k-1]:
            intensity.append(timetrace_sorted[k])
            occurences.append(timetrace_sorted.count(timetrace_sorted[k]))
    return intensity,occurences

def Int_hist(Plot_Int_hist, Filter_histogram, path):
    '''
    This function plot the intensity histogram (if Plot_Int_hist == True) 
    and the filtered curve (if Filter_histogram == True)
    '''

    if Plot_Int_hist == True:

        ## IMPORT DATA ----------------------------------------------------------------------------------------------------------------------------------------------------
        name = name_file(path)
        filename = name + '_int.dat'
        df = import_data(path, filename)
        Timetrace = df['Timetrace']
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------

        occurences       = counts(Timetrace)[1]
        intensity        = counts(Timetrace)[0]
        occurences_model = gaussian_filter1d(occurences, sigma = 5)

        plt.figure(tight_layout=True)
        plt.plot(intensity,occurences)
        if Filter_histogram == True:
            plt.plot(intensity,occurences_model, label = 'gaussian filter')
        plt.xlabel("Intensity")
        plt.ylabel('Occurences')
        plt.title('Photoluminescence intensity histogram')
        plt.grid()
        plt.savefig(path + '{}_intensity_histogram.pdf'.format(name))
        plt.savefig(path + '{}_intensity_histogram.png'.format(name))
        print('Intensity histogram : OK')

def ONLevel_func(Percent, path):
    
    ## IMPORT DATA ----------------------------------------------------------------------------------------------------------------------------------------------------
    name = name_file(path)
    filename = name + '_int.dat'
    df = import_data(path, filename)
    Timetrace = df['Timetrace']
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------

    occurences       = counts(Timetrace)[1]
    intensity        = counts(Timetrace)[0]
    occurences_model = gaussian_filter1d(occurences, sigma = 5)
    on_index         = np.where(occurences_model > 0.01*Percent*np.max(occurences_model))[0][0]
    

    return P[on_index]



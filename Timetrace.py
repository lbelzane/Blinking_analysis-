import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import openpyxl
import os
from data_load import *

def timetrace(Plot_timetrace, Plot_zoom, xmin, xmax, path):

     ## IMPORT DATA ----------------------------------------------------------------------------------------------------------------------------------------------------

    name = name_file(path)
    filename = name + '_int.dat'

    df = import_data(path, filename)
    Timetrace = df['Timetrace']
    tauBlinking = df['tauBlinking']
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------


    ## Timetrace 

    if Plot_timetrace == True:

        figTimeTrace=plt.figure("time trace {}".format(name),figsize=[18,6])
        plt.plot(tauBlinking,Timetrace)
        plt.xlabel("time (s)")
        plt.ylabel("intensity (cps)")
        plt.grid(True,'both')
        plt.tight_layout()
        plt.savefig(path + '{}_timetrace.pdf'.format(name))
        plt.savefig(path + '{}_timetrace.png'.format(name))
        print('Timetrace : OK')

    ## Zoom timetrace

    if Plot_zoom == True:

        fig_zoom = plt.figure("time trace zoom {}".format(name),figsize=[18,6])
        plt.plot(tauBlinking,Timetrace)
        plt.xlabel("time (s)")
        plt.ylabel("intensity (kcps)")
        plt.xlim(xmin,xmax)
        plt.grid()
        plt.tight_layout()
        plt.savefig(path + '{}_timetrace_zoom.pdf'.format(name))
        plt.savefig(path + '{}_timetrace_zoom.png'.format(name))
        print('Zoom Timetrace : OK')




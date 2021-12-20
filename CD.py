import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os 
from CD_functions import *
from data_load import *


def CD(OFF_analysis, ON_analysis, ON_OFF_comp, n, OFFLevel, ONLevel, T_acc, model_power, model_power_exp,path):

    '''
    This function enable us to import data, to establish the cumulative distribution of the ON and OFF
    periods and to save the figures into the data folder.
    '''

    ## IMPORT DATA ----------------------------------------------------------------------------------------------------------------------------------------------------

    name = name_file(path)
    filename = name + '_int.dat'

    df = import_data(path, filename)
    Timetrace = df['Timetrace']
    tauBlinking = df['tauBlinking']
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ### OFF Analysis:

    if OFF_analysis == True: 

        BlinkingOFFCUM = OFF(OFFLevel, n, tauBlinking, Timetrace, df)

        if BlinkingOFFCUM == []:
            print('OFFLevel is too low...')                 
            return None
        else:
            off_zero = np.where(BlinkingOFFCUM == 0)[0][0]

        ## Model 1: Linear regress
        parametres_OFF_1, intercept_OFF, r_OFF, p_OFF, se_OFF = stats.linregress(np.log(tauBlinking[1:off_zero-1]), np.log(BlinkingOFFCUM[1:off_zero-1]))

        ## Model 2: Power exponential law 
        sigma_off = [1/k for k in range(1,off_zero-1)]      # to fit better the last points
        parametres_OFF_2,_ = curve_fit( model_2, tauBlinking[1:off_zero-1], BlinkingOFFCUM[1:off_zero-1], sigma = sigma_off, p0 = [n, 1, T_acc, 1, 0], bounds = ([n-0.0001, 0, T_acc - 0.005, 0.01, -1], [ n, 3, T_acc, 100, 1]) )

        ##Cumulative distribution figure 
        fig_OFF = plt.figure(tight_layout=True)

        if model_power == True:
            plt.plot(tauBlinking, model_1(tauBlinking,parametres_OFF_1, intercept_OFF),label= 'fit')
        if model_power_exp == True:
            plt.plot(tauBlinking, model_2(tauBlinking,*parametres_OFF_2), label= r'fit : $\tau_{0}$ = '+ '{}'.format(round(parametres_OFF_2[3],2)) + r'  $\mu$ = '+ '{}'.format(round(parametres_OFF_2[1],2)))
         
        plt.plot(tauBlinking[1:off_zero-1], BlinkingOFFCUM[1:off_zero-1], '.', label = 'data')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\tau$ (ms)')
        plt.ylabel(r'$P_{off}(\tau_{off}>\tau)$')
        plt.xlim(tauBlinking[1], tauBlinking[off_zero])
        plt.ylim(BlinkingOFFCUM[off_zero-1],np.max(BlinkingOFFCUM))

        plt.grid()
        plt.legend(loc='lower left')
        plt.title('Cumulative distribution OFF periods', fontweight = 'bold')

        plt.savefig(path + '{}_Cumulative_distribution_OFF.png'.format(name))
        plt.savefig(path + '{}_Cumulative_distribution_OFF.pdf'.format(name))

        fichier = open(path + '{}_cumulatif_distribution_OFF.txt'.format(name), 'w')
        fichier.write('\ntau0'+' = {}'.format(parametres_OFF_2[3]))
        fichier.write('\nmu = {}'.format(parametres_OFF_2[1]))
        print('CD OFF : OK')
        
        
    ### ON Analysis: 

    if ON_analysis == True: 

        BlinkingONCUM = ON(ONLevel, n , tauBlinking, Timetrace, df)

        if BlinkingONCUM == []:
            print('ONLevel is too high...')                 
            return None
        else:
            on_zero = np.where(BlinkingONCUM == 0)[0][0]
        
        ## Model 1: Linear regress
        parametres_ON_1, intercept_ON, r_ON, p_ON, se_ON = stats.linregress(np.log(tauBlinking[1:on_zero-1]), np.log(BlinkingONCUM[1:on_zero-1]))

        ## Model 2: Power exponential law 
        sigma_on = [1/k for k in range(1,on_zero-1)]        # to fit better the last points
        parametres_ON_2,_ = curve_fit( model_2, tauBlinking[1:on_zero-1], BlinkingONCUM[1:on_zero-1], sigma = sigma_on, p0 = [n, 1, T_acc, 1, 0], bounds = ([n-0.0001, 0, T_acc - 0.005, 0.01, -1], [ n, 3, T_acc, 10, 2]) )
        
        ##Cumulative distribution figure 
        
        fig_ON = plt.figure(tight_layout=True)

        if model_power == True:
            plt.plot(tauBlinking, model_1(tauBlinking,parametres_ON_1, intercept_ON),label= 'fit')
        if model_power_exp == True:
            plt.plot(tauBlinking, model_2(tauBlinking,*parametres_ON_2), label= r'fit : $\tau_{0}$ = '+ '{}'.format(round(parametres_ON_2[3],2)) + r'  $\mu$ = '+ '{}'.format(round(parametres_ON_2[1],2)))
        plt.plot(tauBlinking[1:on_zero-1], BlinkingONCUM[1:on_zero-1], '.', label = 'data')

        plt.ylim(BlinkingONCUM[on_zero-1],np.max(BlinkingONCUM))
        plt.xlim(tauBlinking[1], tauBlinking[on_zero])
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\tau$ (ms)')
        plt.ylabel(r'$P_{on}(\tau_{on}>\tau)$')
        plt.xlim(tauBlinking[1], tauBlinking[on_zero])
        plt.ylim(BlinkingONCUM[on_zero-1],np.max(BlinkingONCUM))

        plt.grid()
        plt.legend(loc='lower left')
        plt.title('Cumulative distribution ON periods ', fontweight = 'bold')

        plt.savefig(path + '{}_Cumulative_distribution_ON.png'.format(name))
        plt.savefig(path + '{}_Cumulative_distribution_ON.pdf'.format(name))

        fichier = open(path + '{}_cumulatif_distribution_ON.txt'.format(name), 'w')
        fichier.write('\ntau0'+' = {}'.format(parametres_ON_2[3]))
        fichier.write('\nmu = {}'.format(parametres_ON_2[1]))
        print('CD ON : OK')
        
    ### OFF/ON 

    if ON_OFF_comp == True: 

        fig_comp = plt.figure(tight_layout=True)

        plt.plot(tauBlinking[1:off_zero-1], BlinkingOFFCUM[1:off_zero-1], '.', label = 'OFF (OFFLevel = {})'.format(OFFLevel))
        plt.plot(tauBlinking[1:on_zero-1], BlinkingONCUM[1:on_zero-1], '.', label = 'ON (ONLevel = {})'.format(ONLevel), color = 'orange')
        plt.plot(tauBlinking, model_2(tauBlinking,*parametres_OFF_2), label = 'fit', color = 'k')
        plt.plot(tauBlinking, model_2(tauBlinking,*parametres_ON_2), color = 'k')

        plt.xlabel(r'$\tau$ (ms)')
        plt.ylabel(r'$P_{on,off}(\tau_{on,off}>\tau)$')
        plt.xlim(tauBlinking[1], max(tauBlinking[on_zero], tauBlinking[off_zero])+10)
        plt.ylim(min(BlinkingONCUM[on_zero-1],BlinkingOFFCUM[off_zero-1]), np.max(BlinkingONCUM))
        plt.xscale('log')
        plt.yscale('log')

        plt.grid()
        plt.legend()

        plt.savefig(path + '{}_Cumulative_distribution_Comp.png'.format(name))
        plt.savefig(path + '{}_Cumulative_distribution_Comp.pdf'.format(name))
        print('CD ON/OFF : OK')
        


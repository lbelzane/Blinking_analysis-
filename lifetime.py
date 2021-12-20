#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:29:28 2020

@author: pierinis
"""
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
from lifetime_functions import fotone, hist_bin, ExpDecayfit
import os 
from data_load import name_file, import_lifematrix


def create_lifetime_FLID(Create_FLID, Lifetime, Bright_state, Dark_state, Grey_state, OFFLevel, ONLevel, n, path):

    basename     = path[0:-1]
    name = name_file(path)

    metadatafile = path + '{}_metadata.npz'.format(name)
    npyfile      = path + '{}_all.npy'.format(name)
    channel      = 1 #can be 1 or 2
    bin_size     = n #in seconds
    reload_all   = True

    try:
        timetrace
    except NameError:
        reload_all=True
    
    try:
        previous_bin_size
    except NameError:
        previous_bin_size=-1
    
    try:
        previous_channel
    except NameError:
        previous_channel=-1
            
    if reload_all:
        metadata=np.load(metadatafile)
        GlobRes=metadata['GlobRes'][0]
        Resolution=metadata['Resolution'][0]
        data=np.load(npyfile)
        dataT=data.T
    
    
    if (reload_all or bin_size!=previous_bin_size or previous_channel!=channel):
        Dtime             = dataT[2][dataT[0]==channel]
        TimeTag           = dataT[1][dataT[0]==channel]
        Alltime           = np.vstack((TimeTag,Dtime)).T
        previous_bin_size = bin_size
        previous_channel  = channel
        bin_size_int      = int(round(bin_size/GlobRes,ndigits=0))
        bin_total_number  = int(TimeTag[-1]/bin_size_int+1)
        timetrace         = []

        for i in range(bin_total_number):
            timetrace.append(hist_bin(bin_size))
    
        bins = np.arange(start=0,stop=TimeTag[-1]*Resolution,step=bin_size)
        n    = 0

        for element in Alltime:
            f=fotone(element[0],element[1])

            if f.TimeTag<bin_size_int*(n+1):
                timetrace[n].append(f)
            else:
                n+=1
                timetrace[n].append(f)
    
        intensityTrace=[el.getintensity(bin_size) for el in timetrace]
        intensityTrace=np.array(intensityTrace)/1000

    ### Lifetime -----------------------------------------------------------------------------------------------------------------------

    #Lifetime decay: 

    name = name_file(path)
    df_all, df_OFF, df_G, df_ON = import_lifematrix(path, name, OFFLevel, ONLevel)

    figLifetime=plt.figure("lifetime {}".format(name))
    ylifetime,xlifetime=np.histogram(Dtime,bins=np.arange(Dtime.max()+1))
    xlifetime=xlifetime*1e9*Resolution
    x, Fit, p = ExpDecayfit(xlifetime[0:-1],ylifetime, supposeSqrt=True, start_at_from_top=0.95,exp_num=3, x0_vary=False,xmax=350)
    xlifetime = xlifetime-p['x_0']
    plt.semilogy(xlifetime[0:-1],ylifetime/max(ylifetime))
    plt.xlabel('Time (ns)')
    plt.ylabel('Normalized PL Intensity')
    plt.title('Photoluminescence decay', fontsize = 20)
    plt.legend(fontsize = 13)
    plt.grid()
    x=x-p['x_0']

    if Lifetime == True:

        plt.plot(x,Fit/max(Fit),'k','--',label="fit")
        plt.legend()
        plt.xlim([0,350])
        plt.tight_layout()
        plt.savefig(path + '{}_lifetime.png'.format(name))
        plt.savefig(path + '{}_lifetime.pdf'.format(name))
        print(p)
        fichier = open(path + '{}_lifetimefit.txt'.format(name), 'w')
        fichier.write('\ntau1      = {:4.1f}        Amp1      = {:.3f}'.format(p['tau_1'].value,p['Amp_1'].value/max(Fit)))
        fichier.write("\ntau2      = {:4.1f}        Amp2      = {:.3f}".format(p['tau_2'].value,p['Amp_2'].value/max(Fit)))
        fichier.write("\ntau3      = {:4.1f}        Amp3      = {:.3f}".format(p['tau_3'].value,p['Amp_3'].value/max(Fit)))
        print('Lifetime              : OK')
        
        if Bright_state == True: 
            sums_ON     = df_ON.sum()
            T_ON        = sums_ON.axes[0]*10**9
            I_ON        = sums_ON.tolist()
            I_ON_clean  = np.array([I_ON[k] for k in range(len(I_ON)) if I_ON[k]>0])
            T_ON_clean  = np.array([T_ON[k] for k in range(len(T_ON)) if I_ON[k] in I_ON_clean])

            x_ON, Fit_ON, p_ON = ExpDecayfit(T_ON_clean, I_ON_clean, supposeSqrt=True, start_at_from_top=0.95,exp_num=3, x0_vary=False,xmax=350)
            x_ON = x_ON - p_ON['x_0']
            T_ON_clean = T_ON_clean - p_ON['x_0']
            fig_lifetime_ON = plt.figure(tight_layout=True)
            plt.plot(T_ON_clean,I_ON_clean/max(I_ON_clean), 'y', label = 'Bright state')
            plt.plot(x_ON, Fit_ON/max(Fit_ON), '--', color = 'k', zorder=3, label = 'fit')
            plt.yscale('log')
            plt.yscale('log')
            plt.xlabel('Time (ns)')
            plt.ylabel('Normalized PL Intensity')
            plt.xlim(0,358)
            plt.legend(fontsize = 13)
            plt.grid()
            plt.savefig(path + '{}_lifetime_ON.png'.format(name))
            plt.savefig(path + '{}_lifetime_ON.pdf'.format(name))

            fichier = open(path + '{}_lifetimefit.txt'.format(name), 'a')
            fichier.write('\n\ntau_ON_1  = {:4.1f}        Amp_ON_1  = {:.3f}'.format(p_ON['tau_1'].value,p_ON['Amp_1'].value/max(Fit_ON)))
            fichier.write("\ntau_ON_2  = {:4.1f}        Amp_ON_2  = {:.3f}".format(p_ON['tau_2'].value,p_ON['Amp_2'].value/max(Fit_ON)))
            fichier.write("\ntau_ON_3  = {:4.1f}        Amp_ON_3  = {:.3f}".format(p_ON['tau_3'].value,p_ON['Amp_3'].value/max(Fit_ON)))
            print('Lifetime Bright state : OK')

        if Dark_state ==True:
            sums_OFF    = df_OFF.sum()
            T_OFF       = sums_OFF.axes[0]*10**9
            I_OFF       = sums_OFF.tolist()
            I_OFF_clean = np.array([I_OFF[k] for k in range(len(I_OFF)) if I_OFF[k]>0])
            T_OFF_clean = np.array([T_OFF[k] for k in range(len(T_OFF)) if I_OFF[k] in I_OFF_clean])

            x_OFF, Fit_OFF, p_OFF = ExpDecayfit(T_OFF_clean, I_OFF_clean, supposeSqrt=True, start_at_from_top=0.95,exp_num=3, x0_vary=False,xmax= 350)
            x_OFF = x_OFF - p_OFF['x_0']
            T_OFF_clean = T_OFF_clean - p_OFF['x_0']
            fig_lifetime_OFF = plt.figure(tight_layout=True)
            plt.plot(T_OFF_clean,I_OFF_clean/max(I_OFF_clean), 'r', label = 'Dark state')
            plt.plot(x_OFF, Fit_OFF/max(Fit_OFF), '--', color = 'k', label = 'fit')
            plt.yscale('log')
            plt.xlabel('Time (ns)')
            plt.ylabel('Normalized PL Intensity')
            plt.xlim(0,358)
            plt.title('Photoluminescence decay', fontsize = 20)
            plt.legend(fontsize = 13)
            plt.grid()
            plt.savefig(path + '{}_lifetime_OFF.png'.format(name))
            plt.savefig(path + '{}_lifetime_OFF.pdf'.format(name))

            fichier = open(path + '{}_lifetimefit.txt'.format(name), 'a')
            fichier.write('\n\ntau_OFF_1 = {:4.1f}        Amp_OFF_1 = {:.3f}'.format(p_OFF['tau_1'].value,p_OFF['Amp_1'].value/max(Fit_OFF)))
            fichier.write("\ntau_OFF_2 = {:4.1f}        Amp_OFF_2 = {:.3f}".format(p_OFF['tau_2'].value,p_OFF['Amp_2'].value/max(Fit_OFF)))
            fichier.write("\ntau_OFF_3 = {:4.1f}        Amp_OFF_3 = {:.3f}".format(p_OFF['tau_3'].value,p_OFF['Amp_3'].value/max(Fit_OFF)))
            print('Lifetime Dark state   : OK')
            print(p_OFF)

        if Grey_state ==True:
            sums_G      = df_G.sum()
            T_G         = sums_G.axes[0]*10**9
            I_G         = sums_G.tolist()
            I_G_clean   = np.array([I_G[k] for k in range(len(I_G)) if I_G[k]>0])
            T_G_clean   = np.array([T_G[k] for k in range(len(T_G)) if I_G[k] in I_G_clean])

            x_G, Fit_G, p_G = ExpDecayfit(T_G_clean, I_G_clean, supposeSqrt=True, start_at_from_top=0.95,exp_num=3, x0_vary=False,xmax=350)
            x_G = x_G - p_G['x_0']
            T_G_clean = T_G_clean - p_G['x_0']
            fig_lifetime_G = plt.figure(tight_layout=True)
            plt.plot(T_G_clean,I_G_clean/max(I_G_clean), color = "orange", label = 'Grey state')
            plt.plot(x_G, Fit_G/max(Fit_G), '--', color = 'k', label = 'fit')
            plt.yscale('log')
            plt.yscale('log')
            plt.xlabel('Time (ns)')
            plt.ylabel('Normalized PL Intensity')
            plt.xlim(0,358)
            plt.title('Photoluminescence decay', fontsize = 20)
            plt.legend(fontsize = 13)
            plt.grid()
            plt.savefig(path + '{}_lifetime_G.png'.format(name))
            plt.savefig(path + '{}_lifetime_G.pdf'.format(name))

            fichier = open(path + '{}_lifetimefit.txt'.format(name), 'a')
            fichier.write('\n\ntau_G_1   = {:4.1f}        Amp_G_1   = {:.3f}'.format(p_G['tau_1'].value,p_G['Amp_1'].value/max(Fit_G)))
            fichier.write("\ntau_G_2   = {:4.1f}        Amp_G_2   = {:.3f}".format(p_G['tau_2'].value,p_G['Amp_2'].value/max(Fit_G)))
            fichier.write("\ntau_G_3   = {:4.1f}        Amp_G_3   = {:.3f}".format(p_G['tau_3'].value,p_G['Amp_3'].value/max(Fit_G)))
            print('Lifetime Grey state   : OK')

        #Comparaison Intensity/Lifetime:

        meansArr               =   [(element.mean_time(Resolution, 40e-9,120e-9)) for element in timetrace] #era 90 prima
        means_npArr            =   (np.array(meansArr))
        df                     =   pd.DataFrame()
        df["intensity (kcps)"] =   intensityTrace
        df["time (ns)"]        =   means_npArr*1e9-p['x_0']
  
        figTimeTracem, (ax1, ax2)=plt.subplots(2,1, sharex=True)
        ax1.grid()
        ax2.grid()

        ax1.set_xlim(300, 302)
        ax2.set_xlim(300, 302)
        #ax1.set_xlim(55,60)
        #ax2.set_xlim(55,60)
        ax1.plot(np.array(range(bin_total_number))*bin_size,df['intensity (kcps)'])
        ax2.plot(np.array(range(bin_total_number))*bin_size,df["time (ns)"])
        ax2.set_ylim(0,2*np.mean(df["time (ns)"]))
        ax1.set_ylim(0,2*np.mean(df['intensity (kcps)']))
        ax2.set_ylabel("lifetime\n(ns)")
        ax1.set_ylabel("intensity\n(kcps)")
        ax2.set_xlabel("time (s)")
        plt.tight_layout()
        plt.savefig(path + '{}_trace_and_mean_zoom.png'.format(name))
        plt.savefig(path + '{}_trace_and_mean_zoom.pdf'.format(name))

    
    ### FLID -------------------------------------------------------------------------------------------------------------------------------

    if Create_FLID == True:
        
        meansArr               =   [(element.mean_time(Resolution, 40e-9,120e-9)) for element in timetrace] #era 90 prima
        means_npArr            =   (np.array(meansArr))
        df                     =   pd.DataFrame()
        df["intensity (kcps)"] =   intensityTrace
        df["time (ns)"]        =   means_npArr*1e9-p['x_0']
        
        cmap = matplotlib.cm.get_cmap('RdBu_r')
        col = cmap(0)
    
        FigFLID, ax1=plt.subplots(1,1)
        grid=sns.jointplot(x = df["time (ns)"],y = df["intensity (kcps)"], kind='kde',shade=True, n_levels = 100, cmap="RdBu_r", space=0, xlim=(0,35) , ylim=(min(df["intensity (kcps)"]),max(df["intensity (kcps)"])), cbar=False)

        ax= grid.ax_joint
        ax_margx = grid.ax_marg_x
        ax_margy = grid.ax_marg_y

        sns.kdeplot(y = df["intensity (kcps)"], ax = ax_margy, fill = True)
        sns.kdeplot(x = df["time (ns)"], ax = ax_margx, fill = True)
        ax.set_facecolor(col)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)
        grid.space=0
        for c in grid.ax_joint.collections:
            c.set_edgecolor("face")

        grid.savefig(path + '{}_FLID.pdf'.format(name))
        grid.savefig(path + '{}_FLID.png'.format(name))

        #%% confidence level
        H,xedges,yedges = np.histogram2d(df.dropna()["time (ns)"],df.dropna()["intensity (kcps)"],bins=100,normed=True)
        norm = H.sum()
        #print(H)
        #print(norm)
        # Set contour levels
        # contour1=0.90
        contour2=0.68
        contour3=0.50
        
        # Set target levels as percentage of norm
        # target1 = norm*contour1
        target2 = norm*contour2
        target3 = norm*contour3
        
        # Take histogram bin membership as proportional to Likelihood
        # This is true when data comes from a Markovian process
        
        # Find levels by summing histogram to objective
        # level1= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
        level2= 0.5     #scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
        level3= 0.68    #scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))
        level4= 0.9    #H.max()

        #print(H.max())

        #print(level3)
        
        #grid.ax_joint.confidence
        # levels=[level1,level2,level3,level4]
        levels=[level2,level3,level4]
        x_count=(xedges[0:-1]+xedges[1:])/2
        y_count=(yedges[0:-1]+yedges[1:])/2
        
        
        sns.kdeplot(df["time (ns)"],df["intensity (kcps)"], shade=False,ax=ax, n_levels=levels, cmap="Reds_d", norm = plt.Normalize())

        sns.kdeplot(y = df["intensity (kcps)"], ax = ax_margy, fill = True)
        sns.kdeplot(x = df["time (ns)"], ax = ax_margx, fill = True)
        ax.set_facecolor(col)
        grid.savefig(path + '{}_FLID_noted.pdf'.format(name))
        grid.savefig(path + '{}_FLID_noted.png'.format(name))
        print('FLID                  : OK')





#path = '/Users/lucienbelzane/Desktop/Columbia/2021-11-09/em1/'
    

#Lifetime(OFF_state = True, ON_state = True, Grey_state = False, All_state = True, OFFLevel = 60 , ONLevel = 150, path = path)

import numpy as np
import pandas as pd 
from os import listdir

def name_file(path):
    k = 0
    name = ''
    file = listdir(path)[0]
    while file[k] != '_':
        name += file[k]
        k+=1
    return name
        
def import_data(path,filename):

    df = pd.read_csv(path+filename, names = ['Data'])
    df = pd.DataFrame(df.Data.str.split(" ",6).tolist(), columns = ["tauBlinking","Timetrace",'Int_ch1', 'mean_lifetime_ch1','int_ch2','mean_lifetime_ch2'])
    #df = pd.DataFrame(ds.Data.str.split(" ",2).tolist(), columns = ["tauBlinking","Timetrace"])  #Some int.dat files have only two colulmns 
    df = df.drop(0)
    df = df.drop(df.shape[0])
    df = df.astype('float')
    return df

def import_lifematrix(path, filename, OFFLevel, ONLevel):

    filematrix= path + '{}_lifeMatrix.dat'.format(filename)
    datas=np.loadtxt(filematrix)
    df = pd.DataFrame(datas[1:len(datas)], columns=datas[0])
    df_OFF = df[3:int(OFFLevel)]
    df_G = df[int(OFFLevel):int(ONLevel)]
    df_ON = df[int(ONLevel):len(df)]
    return df, df_OFF, df_G, df_ON




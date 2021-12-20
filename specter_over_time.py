
import numpy as np
import pandas as pd
import sys
import os
import spe2py as spe
import spe_loader as sl
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.signal import find_peaks
from os import listdir
from os import path

directory="/Users/lucienbelzane/Desktop/ToLucien/CsPbBr3-justSpectra"


def extractTotalTime(filename):
    pattern=r"\d{4}-\d{2}-\d{2} (?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})"
    for match in re.finditer(pattern, filename):
        h,m,s=int(match.group('hour')),int(match.group('minute')),int(match.group('second'))
        TotalTime=(h*60+m)*60+s
        #print(TotalTime)
        print("h={},m={},s={}".format(h,m,s))
    
    return TotalTime
#create a list with the files in that directory

onlyfiles = [f for f in listdir(directory) if path.isfile(path.join(directory, f))]
onlyfiles = [file for file in onlyfiles if file[-4:]=='.spe']
TotalTime= [extractTotalTime(f) for f in onlyfiles]

ds=pd.DataFrame({'file_names':onlyfiles, 'time':TotalTime})
ds['red_time']=ds['time']-min(ds['time'])
#ds['wavelength']=0;
#ds['data']=0;
#ds['wavelenght_emission']=0
#ds['peak_width']=0
plt.ioff()

class dataClass:
    def __init__(self, v):
        self.v = v

X=np.array([])
Y=np.array([])
Z=np.array([])
Zn=np.array([])
for i, row in ds.iterrows():
#    path=line['wave_file']
    nomefile=row.file_names

    path=os.path.join(directory,nomefile)
    path_fig=os.path.splitext(path)[0]+'_spectrum.png'
#%
    #print(path)
    #print(nomefile,'\n');
    spe_file=sl.load_from_files([path])
    x_min=int(spe_file.roi[0]['x'])
    x_max=int(spe_file.roi[0]['x'])+int(spe_file.roi[0]['width'])
    wavelength=np.array(spe_file.wavelength[x_min:x_max])
    
    fig=plt.figure()
    data=spe_file.data[0][0][0]
    plt.rcParams.update({'font.size': 22})
    plt.plot(wavelength, data)
    plt.xlabel('wavelength (nm)')
    plt.ylabel('intensity (A.U.)')
    plt.tight_layout()
    plt.grid(True)
    
    datan=data/data.max()
    
    peaks, properties=find_peaks(datan,prominence=0.2,width=4)
    if len(peaks)!=1:
        print("Error in peak detecting in file; ", nomefile,"\n")
  #         raise Exception("Error in detecting peaks")
    else:
        peak=peaks[0]
    
    l=spe_file.wavelength[peak]
    h=data[peak]
    delta=spe_file.wavelength[1]-spe_file.wavelength[0]
    width=properties['widths'][0]*delta
    ds.loc[i,"wavelenght_emission"]=l
    ds.loc[i,"peak_width"]=width
    w=dataClass(wavelength)
    d=dataClass(data)
    ds.loc[i,"wavelength"]=w
    ds.loc[i,"data"]=d
    for wave, dato in zip(wavelength,data):
        X=np.append(X,wave)
        Y=np.append(Y,row.red_time)
        Z=np.append(Z,dato)
    Zn=np.append(Zn,data/data.max())
#    plt.vlines(l,0,h)
    plt.tight_layout()
    plt.savefig(path_fig,format='png')
    plt.close(fig)
    fig.clear()
       # plt.show()
plt.ion()

# fig=plt.figure()
# X1=wavelength
# Y1=ds.red_time
# Z1=np.reshape(Zn,(len(Y1),len(X1)))
# graph=plt.contourf(X1,Y1/60,Z1,levels=100,extend='both')
# plt.xlim(480,560)
# plt.tight_layout()
# plt.show()

fig=plt.figure()
cnt=plt.tricontourf(X,Y/60,Zn,levels=100)
plt.xlim(490,540)
plt.xlabel("wavelength (nm)")
plt.ylabel('time (min)')
plt.tight_layout()
# This is the fix for the white lines between contour levels
for c in cnt.collections:
    c.set_edgecolor("face")
col=plt.colorbar()
col.set_ticks([0,0.2,0.4,0.6,0.8,1])
plt.savefig(os.path.join(directory,"spectra_degration_n.pdf"))
plt.savefig(os.path.join(directory,"spectra_degration_n.png"))

fig=plt.figure()
plt.tricontourf(X,Y/60,Z,levels=100)
plt.xlim(490,540)
plt.xlabel("wavelength (nm)")
plt.ylabel('time (min)')
plt.tight_layout()
plt.colorbar()
plt.savefig(os.path.join(directory,"spectra_degration.pdf"))
plt.savefig(os.path.join(directory,"spectra_degration.png"))




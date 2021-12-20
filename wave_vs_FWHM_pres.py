
import numpy as np
import pandas as pd
import sys
import os
import re #for regular expressions
import spe2py as spe
import spe_loader as sl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import matplotlib

directory='/Users/lucienbelzane/Desktop/Columbia/Columbia_spectra/allspectra'
pattern=r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2}) (?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})"

def get_date(filename):
    match=re.search(pattern,filename)[0]
    y,m,d=int(match.group('year')),int(match.group('month')),int(match.group('day'))
    h,minutes,s=int(match.group('hour')),int(match.group('minute')),int(match.group('second'))
    date=pd.Timestamp(year=y,month=m,day=d,hour=h,minute=minutes,second=s)
    return date

def get_spectra_analysis(filename):
    match=re.search(pattern,filename)
    y,m,d=int(match.group('year')),int(match.group('month')),int(match.group('day'))
    print(m)
    h,minutes,s=int(match.group('hour')),int(match.group('minute')),int(match.group('second'))
    date=pd.Timestamp(year=y,month=m,day=d,hour=h,minute=minutes,second=s)
    
    filename=os.path.join(directory,filename)
    path=os.path.abspath(filename)
    basename=filename
    path_fig=os.path.abspath(basename)
    path_fig=os.path.splitext(path_fig)[0]+'_spectrum.pdf'
    
    spe_file=sl.load_from_files([path])
    x_min=int(spe_file.roi[0]['x'])
    x_max=int(spe_file.roi[0]['x'])+int(spe_file.roi[0]['width'])
    wavelength=np.array(spe_file.wavelength[x_min:x_max])
    
    fig=plt.figure()
    data=spe_file.data[0][0][0][1:len(wavelength)]
    plt.rcParams.update({'font.size': 22})
    plt.plot(wavelength[1:len(wavelength)], data)
    plt.xlabel('wavelength (nm)')
    plt.ylabel('intensity (A.U.)')
    plt.tight_layout()
    plt.grid(True)
    
    datan=data/data.max()

    peaks, properties=find_peaks(datan,prominence=0.8,width=4, height=0.9, distance=20)
    
    if len(peaks)!=1:
        print("Error in peak detecting in file; ", filename,"\n{} peaks found:\n{}".format(len(peaks),peaks))
        peak=int(np.array(peaks).mean())
        #raise Exception("Error in detecting peaks")
    else:
        peak=peaks[0]
    
    l=spe_file.wavelength[peak]
    h=data[peak]
    delta=spe_file.wavelength[1]-spe_file.wavelength[0]
    width=properties['widths'][0]*delta
    if width>40:
        print(filename)
    #dataset.loc[i,"wavelenght_emission"]=l
    #dataset.loc[i,"peak_width"]=width
    #plt.vlines(l,0,h)
    plt.savefig(path_fig,format='pdf')
    plt.close(fig)
    fig.clear()
    return l,width

df=pd.DataFrame()


df["file_name"]=[file for file in os.listdir(directory) if file.endswith('.spe')]


#col=df["file_name"].apply(get_spectra_analysis)

df[['wavelenght', 'FWHM']] = pd.DataFrame(zip(*df['file_name'].apply(get_spectra_analysis))).transpose()


#df[['wavelenght','FWHM']]=pd.DataFrame(col.tolist(),index=df.index())



#%% wavelenght distribution
#plt.hist(dataset['wavelenght_emission'][dataset['wavelenght_emission']!=0],30)
fig=plt.figure()
#sb=sns.violinplot(dataset['wavelenght_emission'][dataset['wavelenght_emission']!=0])
sns.kdeplot(df['wavelenght'][df['wavelenght']!=0], shade=True, legend=False, cbar=True)
plt.title(r"density distribution of $\lambda$")
plt.xlabel('wavelenght (nm)')
plt.ylabel('Occurences')
plt.tight_layout()
plt.savefig('/Users/lucienbelzane/Desktop/Columbia/Columbia_spectra/allspectra/lambda_density_dist.pdf',format='pdf')



#%% wavelenght vs FWHM kdeplot
cmap = matplotlib.cm.get_cmap('Blues')
col = cmap(0)
we=df['wavelenght']
pw=df['FWHM']
fig=sns.jointplot(we,pw, kind='kde', shade = True, cmap = 'Blues', ratio=3, space=0, n_levels = 50)
ax= fig.ax_joint
ax.set_facecolor(col)
#fig=sns.kdeplot(we,pw,shade=True)
#fig.plot_marginals(sns.kdeplot,shade='true')
fig.set_axis_labels(xlabel="wavelength (nm)", ylabel="peak FWHM (nm)")
#fig.ax_joint.set_xlim(470,530)
fig.savefig(os.path.join(directory,"wave_vs_FWHM_p.pdf"))
fig.savefig(os.path.join(directory,"wave_vs_FWHM_p.png"))
#%%


# #%%
# csv_file='/home/damatom/Documenti/Dottorato/DatiParigi/perokytes/PL4/dati.csv'
# xlsx_file='/home/damatom/Documenti/Dottorato/DatiParigi/perokytes/PL4/Appunti_Analisi.xlsx'
# #dataset = pd.read_csv(csv_file,sep=',')
# dataset = pd.read_excel(xlsx_file)
# dataset = dataset[dataset['wave_file'].notnull()]
# dataset.insert(len(dataset.columns),"wavelenght_emission",0.)
# dataset.insert(len(dataset.columns),"peak_width",0.)

# #%%
# plt.ioff()
# for i, row in dataset.iterrows():
# #    path=line['wave_file']
#     nomefile=row.wave_file
#     basename=row["file"]
#     if nomefile=='1' or nomefile==1:
#         continue
#     else: 
#         print(nomefile)

#     #nomefile=path
#         path=os.path.abspath(nomefile)
#         path_fig=os.path.abspath(basename)
#         path_fig=os.path.splitext(path_fig)[0]+'_spectrum.pdf'
# #%
#         #print(path)
#         #print(nomefile,'\n');
#         spe_file=sl.load_from_files([path])
#         x_min=int(spe_file.roi[0]['x'])
#         x_max=int(spe_file.roi[0]['x'])+int(spe_file.roi[0]['width'])
#         wavelength=np.array(spe_file.wavelength[x_min:x_max])
        
#         fig=plt.figure()
#         data=spe_file.data[0][0][0]
#         plt.rcParams.update({'font.size': 22})
#         plt.plot(wavelength, data)
#         plt.xlabel('wavelength (nm)')
#         plt.ylabel('intensity (A.U.)')
#         plt.tight_layout()
#         plt.grid(True)
        
#         datan=data/data.max()
        
#         peaks, properties=find_peaks(datan,prominence=0.8,width=4)
#         if len(peaks)!=1:
#             print("Error in peak detecting in file; ", nomefile,"\n")
#   #         raise Exception("Error in detecting peaks")
#         else:
#             peak=peaks[0]
        
#         l=spe_file.wavelength[peak]
#         h=data[peak]
#         delta=spe_file.wavelength[1]-spe_file.wavelength[0]
#         width=properties['widths'][0]*delta
#         dataset.loc[i,"wavelenght_emission"]=l
#         dataset.loc[i,"peak_width"]=width
#         plt.vlines(l,0,h)
#         plt.savefig(path_fig,format='pdf')
#         plt.close(fig)
#         fig.clear()
#        # plt.show()
# plt.ion()
# dataset.to_excel(os.path.splitext(csv_file)[0]+'_done.xlsx')

# #%% wavelenght distribution
# #plt.hist(dataset['wavelenght_emission'][dataset['wavelenght_emission']!=0],30)
# fig=plt.figure()
# #sb=sns.violinplot(dataset['wavelenght_emission'][dataset['wavelenght_emission']!=0])
# sns.kdeplot(dataset['wavelenght_emission'][dataset['wavelenght_emission']!=0], shade=True, legend=False, cbar=True)
# plt.title("density distribution of $\lambda$")
# plt.xlabel('wavelenght (nm)')
# plt.ylabel('intensity (A.U.)')
# plt.tight_layout()
# plt.savefig('/home/damatom/Documenti/Dottorato/DatiParigi/perokytes/PL4/lambda_density_dist.pdf',format='pdf')

# plt.show()


# #%% g2 statistic
# fig=plt.figure()
# plt.title("$g^{(2)}(0)$ as function of $\lambda$")
# plt.xlabel("wavelength (nm)")
# plt.ylabel("$g^{(2)}(0)$")
#ds=dataset[dataset['wavelenght_emission'].astype('float')>0]
# plt.plot(ds['wavelenght_emission'],ds['G2(0) clean'].astype('float'), ls='', marker='.')
# plt.rcParams.update({'font.size': 20})

# plt.tight_layout()
# plt.savefig('/home/damatom/Documenti/Dottorato/DatiParigi/perokytes/PL4/g2_vs_lambda.pdf',format='pdf')
# plt.show()

# #ds_g2=ds[ds['I Sat'].notna() and if ds['I Meas'].notna()]
# ds_g2=ds[(ds['I Sat'].notna() & ds['I Meas'].isna()) | (ds['I Sat'].notna() & ds['I Sat']==ds['I Meas'])]

# fig=plt.figure()
# #plt.title("$g^{2}(0)$ as function of $\lambda$")
# plt.xlabel("wavelength (nm)")
# plt.ylabel("$g^{(2)}(0)$")
# #ds=dataset[dataset['wavelenght_emission'].astype('float')>0]
# plt.plot(ds_g2['wavelenght_emission'],ds_g2['G2(0) clean'].astype('float'), ls='', marker='o')
# plt.rcParams.update({'font.size': 20})

# plt.tight_layout()
# #plt.savefig('/home/damatom/Documenti/Dottorato/DatiParigi/perokytes/PL4/g2_vs_lambda_psat.pdf',format='pdf')
# plt.savefig('/home/damatom/Documenti/Dottorato/DatiParigi/perokytes/PL4/FigureArticolo/Fig6_g2_vs_lambda_psat.pdf',format='pdf')

# plt.show()
####%%
#we=ds['wavelenght_emission']
#pw=ds['peak_width']
#fig=sns.jointplot(we,pw, kind='kde', ratio=3, space=0)
#fig.plot_marginals(sns.kdeplot,shade='true')
#fig.set_axis_labels(xlabel="wavelength (nm)", ylabel="peak FWHM (nm)")
#%% Saturation statistics
# import matplotlib.pyplot as plt
# fig=plt.figure()
# plt.hist(ds[ds['I Sat']<100e-7]['I Sat']*1e6, bins=40)
# ds['I Sat'].dtype(np.float32)
# ds['I Sat'].var


'''
THE FOLLOWING IS AN OTHER SCRIPT FOR THE SATURATION

#%% Cacoli intensità
# I=P*(4E5)*(5.47E-6/(405e-9*0.13E-3)^2)
focale_lentina=11e-3
NA=1.4
NAf=0.12
alpha=2*NAf
D=alpha*focale_lentina
#D=(NAf*2)*focale_lentina #diametro del laser
A=(np.pi/4)*(D**2)
lamda=405e-9
WD=0.13E-3
#%% Saturation graph
#4E5 è dato da 50ps di impulso ogni 200ns
fattore=(4E3)*(A/(lamda*WD)**2)
path_dir='/home/damatom/Documenti/Dottorato/DatiParigi/perokytes/PL4/2020_02_28/15h49/'
Exc_pow=np.load(path_dir+'Sin_Psat.npy')
S_intensity=np.load(path_dir+'Sout_Psat.npy')
param=np.load(path_dir+'par.npy')
#A*((1-np.exp(-(x-x_0)/Psat)) + B*(x-x_0)) + C
pdict={'Amp':param[0],
       'Sat':param[1],
       'B':param[2],
       'Offset':param[3],
       'x_0':param[4]}
x=np.arange(0,Exc_pow.max(),Exc_pow.max()/1000)
func=pdict['Amp']*((1-np.exp(-(x-pdict['x_0'])/pdict['Sat'])) + pdict['B']*(x-pdict['x_0']))+pdict['Offset']

fig_sat=plt.figure()
plt.plot(Exc_pow*1e9,S_intensity/pdict['Amp'],ls='', marker='o')
plt.plot(x*1e9,func/pdict['Amp'])
plt.xlabel('power (nW)')
plt.ylabel('intensty')
plt.tight_layout()
plt.savefig(path_dir+'sat_pow.pdf')

fig_sat2=plt.figure()
y_sat=pdict['Amp']*((1-np.exp(-(x-pdict['x_0'])/pdict['Sat'])))+pdict['Offset']
y_nsat=pdict['Amp']*pdict['B']*(x-pdict['x_0'])+pdict['Offset']
plt.plot(Exc_pow*1e9,S_intensity/pdict['Amp'],ls='', marker='o')
plt.plot(x*1e9,func/pdict['Amp'])
plt.plot(x*1e9,y_sat)
plt.plot(x*1e9,y_nsat)
plt.xlabel('power (nW)')
plt.ylabel('intensty')
plt.tight_layout()
plt.savefig(path_dir+'sat_pow_all.pdf')
plt.show()

fig_sat=plt.figure()
fattore*=(1e-6)**2
plt.plot(Exc_pow*fattore,S_intensity/pdict['Amp'],ls='', marker='o')
plt.plot(x*fattore,func/pdict['Amp'])
plt.xlabel('intensity ($W/\mu m^2$)')
plt.ylabel('intensty')
plt.tight_layout()
plt.savefig(path_dir+'sat_intensity.pdf')
plt.show()
'''
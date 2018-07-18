
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
import pylab

#I prefer executing this as a script rather than a function for debugging purposes

#I thought it would be better to make this a variable -- had to take out one of the SN-GRBs and it took ages to fix this! 
no_GRBs = 10
DOTESTS = True

#all the input files
platefitfile = 'platefitflux_072117.txt'
SNinfofile = 'SNinfo_072617.txt'
GRBfile = 'GRBflux_083017_shan_astro9.txt'
SDSSfile = 'sdss_meas.txt'
SDSSerr = 'sdss_err.txt'
SLSNfile = 'SLSN_i_flux.txt'

sncolors = {"Ic": ('black','dotted'),
            "Ic-BL":('#386890', 'dashed'),
            "GRB":('IndianRed','solid'),
            "sdss":('#FFD700', "dashed"),
            "slsn":('#9400D3',"solid")}
            
#def BPT(platefitfile, SNinfofile, GRBfile):

names_slsn=['name', '[OII]3727', 'e_[OII]3727','Hb','e_Hb','[OIII]4959','e_[OIII]4959','l_[OIII]5007','[OIII]5007','e_[OIII]5007','l_Ha','Ha','e_Ha','l_[NII]6584','[NII]6584','e_[NII]6584','[NII]6548','e_[NII]6548','[SII]6717','e_[SII]6717','[SII]6731','e_[SII]6731']   
slsn = pd.read_csv(SLSNfile,skiprows=1, delim_whitespace=True, header=None, names=names_slsn)       

#---------------------Compute SLSN line ratios ----------------------------------------------------------------------------------------

#BPT values for SDSS galaxies
#let's exclude the galaxies that are upper limits in [OIII]5007, [NII]6584 and Ha
#so below are the indices where _no_ flags are
ind_o3 = list(np.where(slsn['l_[OIII]5007'] != '<')[0])
ind_n2 = list(np.where(slsn['l_[NII]6584'] != '<')[0])
ind_ha = list(np.where(slsn['l_Ha'] != '<')[0])
#now merge them and find only the indeces that all of them have:
intermediate = list(set(ind_o3).intersection(ind_n2))
ind_noflags = list(set(intermediate).intersection(ind_ha))

#include a switch that can be used to automatically include or exclude flags:
#e.g. when noflags = 'True', then we exclude the flags and vice versa
noflags = 'False'

'''
slsn['[OIII]5007'] = slsn['[OIII]5007'][ind_noflags]
slsn['e_[OIII]5007'] = slsn['e_[OIII]5007'][ind_noflags]
slsn['[NII]6584'] = slsn['[NII]6584'][ind_noflags]
slsn['e_[NII]6584'] = slsn['e_[NII]6584'][ind_noflags]
slsn['Hb'] = slsn['Hb'][ind_noflags]
slsn['e_Hb'] = slsn['e_Hb'][ind_noflags]
slsn['Ha'] = slsn['Ha'][ind_noflags]
slsn['e_Ha'] = slsn['e_Ha'][ind_noflags]'''

slsn['logOIII_Hb'] = np.log10(slsn['[OIII]5007']/slsn['Hb'])
slsn['logNII_Ha'] = np.log10(slsn['[NII]6584']/slsn['Ha'])

#errors 
slsn['elogOIII'] = slsn['e_[OIII]5007']/(slsn['[OIII]5007']*np.log(10.))
slsn['elogHb'] = slsn['e_Hb']/(slsn['Hb']*np.log(10.))
slsn['elogOIII_Hb'] = np.sqrt(slsn['elogOIII']**2.+slsn['elogHb']**2.)

slsn['elogNII'] = slsn['e_[NII]6584']/(slsn['[NII]6584']*np.log(10.))
slsn['elogHa'] = slsn['e_Ha']/(slsn['Ha']*np.log(10.))
slsn['elogNII_Ha'] = np.sqrt(slsn['elogNII']**2.+slsn['elogHa']**2.)

#---------------------Compute SLSN line ratios ----------------------------------------------------------------------------------------

#---------------------Compute SDSS line ratios ----------------------------------------------------------------------------------------

sdss = pd.read_csv(SDSSfile, skiprows=1, delim_whitespace=True, header=None, index_col='galnum', \
names=['galnum', '[OII]3727','Hb','[OIII]4959','[OIII]5007','[OI]6300','Ha','[NII]6584','[SII]6717','[SII]6731','[SIII]9069','[SIII]9532'])
sdss_err = pd.read_csv(SDSSerr, skiprows=1, delim_whitespace=True, header=None, index_col='galnum', \
names=['galnum', '[OII]3727','Hb','[OIII]4959','[OIII]5007','[OI]6300','Ha','[NII]6584','[SII]6717','[SII]6731','[SIII]9069','[SIII]9532'])

#BPT values for SDSS galaxies
sdss['logOIII_Hb'] = np.log10(sdss['[OIII]5007']/sdss['Hb'])
sdss['logNII_Ha'] = np.log10(sdss['[NII]6584']/sdss['Ha'])

sdss['elogOIII'] = sdss_err['[OIII]5007']/(sdss['[OIII]5007']*np.log(10.))
sdss['elogHb'] = sdss_err['Hb']/(sdss['Hb']*np.log(10.))
sdss['elogOIII_Hb'] = np.sqrt(sdss['elogOIII']**2.+sdss['elogHb']**2.)

sdss['elogNII'] = sdss_err['[NII]6584']/(sdss['[NII]6584']*np.log(10.))
sdss['elogHa'] = sdss_err['Ha']/(sdss['Ha']*np.log(10.))
sdss['elogNII_Ha'] = np.sqrt(sdss['elogNII']**2.+sdss['elogHa']**2.)

#---------------------Compute SDSS line ratios ----------------------------------------------------------------------------------------

PTFs = pd.read_csv(platefitfile, skiprows=9, delim_whitespace=True, header=None, index_col='galnum', \
names=['galnum', 'subdir', 'PTFname_and_region', \
'f_OII', 'ef_OII', 'f_Hr', 'ef_Hr', 'f_Hb', 'ef_Hb', 'f_OIII0', 'ef_OIII0', 'f_OIII', 'ef_OIII', \
'f_Ha', 'ef_Ha', 'f_NII', 'ef_NII', 'f_SII0', 'ef_SII0', 'f_SII1', 'ef_SII1'])

SNinfos = pd.read_csv(SNinfofile, skiprows=4, delim_whitespace=True, header=None, index_col='galnum', \
names=['galnum', 'subdir', 'SNtype'])

names_regions = PTFs['PTFname_and_region'].str.split('-')
PTFs['PTFname'] = names_regions.str[0]
PTFs['region'] = names_regions.str[2]

PTFs = pd.merge(PTFs, SNinfos, left_index=True, right_index=True)
PTFs = PTFs[PTFs['SNtype'] != 'other']

print('Total number of gal: ', PTFs['PTFname'].count())

#GRBs = pd.read_csv(GRBfile, skiprows=2, nrows=9, delim_whitespace=True, header=None, index_col='galnum', \
#GRBs = pd.read_csv(GRBfile, skiprows=2, nrows=10, delim_whitespace=True, header=None, index_col='galnum', \
GRBs = pd.read_csv(GRBfile, skiprows=2, nrows=no_GRBs, delim_whitespace=True, header=None, index_col='galnum', \
names=['galnum', 'paper', 'PTFname_and_region', \
'f_OII', 'ef_OII', 'f_Hr', 'ef_Hr', 'f_Hb', 'ef_Hb', 'f_OIII0', 'ef_OIII0', 'f_OIII', 'ef_OIII', \
'f_Ha', 'ef_Ha', 'f_NII', 'ef_NII', 'f_SII0', 'ef_SII0', 'f_SII1', 'ef_SII1'])

GRBs['SNtype'] = 'GRB'
GRBs['region'] = 'SNsite'
# plot 2003lw with a different symbol
# print GRBs.loc[GRBs['PTFname_and_region'] == 'GRB031203/SN2003lw-host', ['PTFname_and_region', 'region']]
# GRBs['region'][GRBs['PTFname_and_region'] == 'GRB031203/SN2003lw-host'] = 'other'
GRBs.loc[GRBs['PTFname_and_region'] == 'GRB031203/SN2003lw-host', 'region'] = 'other'


# plot 2009nz with a different symbol
#GRBs.loc[GRBs['PTFname_and_region'] == 'GRB091127/SN2009nz-host', 'region'] = 'other'
PTFs = pd.concat([PTFs, GRBs], ignore_index=True)

#Replace any '...' in file with 'NaN' -- getting errors in computations below where missing values are denoted as '...' in .txt files
#PTFs['f_Hb'] = PTFs['f_Hb'].replace('...', float('nan'), regex=True)
#PTFs['f_NII'] = PTFs['f_NII'].replace('...', float('nan'), regex=True)
#PTFs['f_Ha'] = PTFs['f_Ha'].replace('...', float('nan'), regex=True)

# Make BPT diagram
PTFs['logOIII_Hb'] = np.log10(PTFs['f_OIII']/PTFs['f_Hb'])
PTFs['logNII_Ha'] = np.log10(PTFs['f_NII']/PTFs['f_Ha'])

PTFs['elogOIII'] = PTFs['ef_OIII']/(PTFs['f_OIII']*np.log(10.))
PTFs['elogHb'] = PTFs['ef_Hb']/(PTFs['f_Hb']*np.log(10.))

#got issues with the dtype of those arrays because as soon as an entry was 'NaN', the dtype turned from 'float64' into 'object'
b = PTFs['elogHb']**2.
b = pd.Series(b.convert_objects(convert_numeric = True))

PTFs['elogOIII_Hb'] = np.sqrt(PTFs['elogOIII']**2.+b)

PTFs['elogNII'] = PTFs['ef_NII']/(PTFs['f_NII']*np.log(10.))
PTFs['elogHa'] = PTFs['ef_Ha']/(PTFs['f_Ha']*np.log(10.))

#got issues with the dtype of those arrays because as soon as an entry was 'NaN', the dtype turned from 'float64' into 'object'
cc = PTFs['elogHa']**2.
cc = pd.Series(cc.convert_objects(convert_numeric = True))
dd = PTFs['elogNII']**2.
dd = pd.Series(dd.convert_objects(convert_numeric = True))

PTFs['elogNII_Ha'] = np.sqrt(dd+cc)

PTFs.loc[PTFs['PTFname_and_region'] == 'GRB091127/SN2009nz-host', 'elogNII_Ha'] = 0.
PTFs.loc[PTFs['PTFname_and_region'] == 'GRB130702A/SN2013dx-host', 'elogNII_Ha'] = 0.

print PTFs.loc[PTFs['PTFname_and_region'] == 'GRB130702A/SN2013dx-host']


PTFs = PTFs.drop(['elogOIII', 'elogHb', 'elogNII', 'elogHa'], axis=1)



###########plot
#changed figure size from (10,10) to (12,10) to fit the ylabel
fig = plt.figure(figsize=(12, 10))
gspec = gridspec.GridSpec(4, 4)
gspec.update(wspace=0, hspace=0)

top_histogram = plt.subplot(gspec[0, 0:3])
side_histogram = plt.subplot(gspec[1:, 3])
lower_left = plt.subplot(gspec[1:, 0:3])

fs = 20
x_range = [-2.4, 0.0]
y_range = [-1.2, 1]

#plt.title('BPT Diagram', fontsize=fs)
plt.xlabel(r'$\log\ {\rm [NII]_{6584}/H\alpha}$', fontsize=fs)
plt.ylabel(r'$\log\ {\rm [OIII]_{5007}/H\beta}$', fontsize=fs)
#ax = fig.add_subplot(111)
plt.tick_params(axis='both', which='major', labelsize=fs-5, direction="in")
plt.minorticks_on()
plt.xlim(x_range[0], x_range[1])
plt.ylim(y_range[0], y_range[1])

lx = np.arange(1350)*0.001-1.5
ly=0.61/(lx-0.05)+1.3
lower_left.plot(lx, ly, 'k', linewidth=2.5, color='grey')

lx = np.arange(1500)*0.001-1.5
ly=0.61/(lx-0.47)+1.19
lower_left.plot(lx, ly, 'k--', linewidth=2.5, color='grey')

#----Plot SDSS line ratios ---------------------------------------------------------------------------------------------------------

l_sdss = lower_left.errorbar(sdss['logNII_Ha'], sdss['logOIII_Hb'], \
xerr=sdss['elogNII_Ha'], yerr=sdss['elogOIII_Hb'], \
fmt='o', color=sncolors['sdss'][0], alpha = 0.1, label='SDSS galaxies', ms=10, mew=1.5) #markeredgecolor='none')

#label='SDSS galaxies', 

#----Plot SDSS line ratios ---------------------------------------------------------------------------------------------------------

#----Plot SLSN line ratios ---------------------------------------------------------------------------------------------------------
if noflags == 'False':
    l_slsn = lower_left.errorbar(slsn['logNII_Ha'], slsn['logOIII_Hb'], \
    xerr=slsn['elogNII_Ha'], yerr=slsn['elogOIII_Hb'], \
    fmt='o', color=sncolors['slsn'][0], label='SLSN Ic host galaxies', ms=10, mew=1.5) #markeredgecolor='none')
if noflags == 'True':
    #below plots only the slsn galaxies that did not have any flags
    l_slsn = lower_left.errorbar(slsn['logNII_Ha'][ind_noflags], slsn['logOIII_Hb'][ind_noflags], \
    xerr=slsn['elogNII_Ha'][ind_noflags], yerr=slsn['elogOIII_Hb'][ind_noflags], \
    fmt='o', color=sncolors['slsn'][0], label='SLSN Ic host galaxies', ms=10, mew=1.5) #markeredgecolor='none')


label_slsn='SLSN Ic'

#----Plot SLSN line ratios ---------------------------------------------------------------------------------------------------------


l1 = lower_left.errorbar(PTFs['logNII_Ha'][(PTFs['SNtype'] == 'Ic') & (PTFs['region'] != "SNsite")], PTFs['logOIII_Hb'][(PTFs['SNtype'] == 'Ic') & (PTFs['region'] != "SNsite")], \
xerr=PTFs['elogNII_Ha'][(PTFs['SNtype'] == 'Ic') & (PTFs['region'] != "SNsite")], yerr=PTFs['elogOIII_Hb'][(PTFs['SNtype'] == 'Ic') & (PTFs['region'] != "SNsite")], \
fmt='ks', fillstyle='none', label='SN Ic (nuc or nearby HII)', mew=1.5, ms=10)

ll1 = 'SN Ic (nuc or nearby HII)'

l2 = lower_left.errorbar(PTFs['logNII_Ha'][(PTFs['SNtype'] == 'Ic') & (PTFs['region'] == "SNsite")], PTFs['logOIII_Hb'][(PTFs['SNtype'] == 'Ic') & (PTFs['region'] == "SNsite")], \
xerr=PTFs['elogNII_Ha'][(PTFs['SNtype'] == 'Ic') & (PTFs['region'] == "SNsite")], yerr=PTFs['elogOIII_Hb'][(PTFs['SNtype'] == 'Ic') & (PTFs['region'] == "SNsite")], \
fmt='ks', label='SN Ic (SN site)', mew=1.5, ms=10)

ll2 = 'SN Ic (SN site)'

l3 = lower_left.errorbar(PTFs['logNII_Ha'][(PTFs['SNtype'] == 'Ic-BL') &
                                           (PTFs['region'] != "SNsite")],
                         PTFs['logOIII_Hb'][(PTFs['SNtype'] == 'Ic-BL') &
                                            (PTFs['region'] != "SNsite")],
                         xerr=PTFs['elogNII_Ha'][(PTFs['SNtype'] == 'Ic-BL') &
                                                 (PTFs['region'] != "SNsite")],
                         yerr=PTFs['elogOIII_Hb'][(PTFs['SNtype'] == 'Ic-BL') &
                                                  (PTFs['region'] != "SNsite")],
                         fmt='o', color='#386890', fillstyle='none',
                         label='SN Ic-BL (nuc or nearby HII)', mew=1.5, ms=10)

       
ll3 = 'SN Ic-BL (nuc or nearby HII)'

l4 = lower_left.errorbar(PTFs['logNII_Ha'][(PTFs['SNtype'] == 'Ic-BL') &
                                           (PTFs['region'] == "SNsite")],
                         PTFs['logOIII_Hb'][(PTFs['SNtype'] == 'Ic-BL') &
                                            (PTFs['region'] == "SNsite")],
                         xerr=PTFs['elogNII_Ha'][(PTFs['SNtype'] == 'Ic-BL') &
                                                 (PTFs['region'] == "SNsite")],
                         yerr=PTFs['elogOIII_Hb'][(PTFs['SNtype'] == 'Ic-BL') &
                                                  (PTFs['region'] == "SNsite")],
                         fmt='o', color = '#386890', label='SN Ic-BL (SN site)',
                         ms=10, mew=1.5, markeredgecolor='none')

ll4 = 'SN Ic-BL (SN site)'

#print PTFs[['elogNII_Ha', 'elogOIII_Hb', 'logNII_Ha', 'logOIII_Hb', 'PTFname_and_region', 'region']][(PTFs['SNtype'] == 'GRB')]
tempx =  PTFs['logNII_Ha'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] != "SNsite")].tolist()
tempy =  PTFs['logOIII_Hb'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] != "SNsite")].tolist()
tempxerr = PTFs['elogNII_Ha'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] != "SNsite")].tolist()
tempyerr = PTFs['elogOIII_Hb'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] != "SNsite")].tolist()

#lower_left.errorbar(PTFs['logNII_Ha'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] != "SNsite")], PTFs['logOIII_Hb'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] != "SNsite")], \
#xerr=PTFs['elogNII_Ha'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] != "SNsite")], yerr=PTFs['elogOIII_Hb'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] != "SNsite")], \
l5 = lower_left.errorbar(tempx, tempy, xerr=tempxerr, yerr=tempyerr, fmt='D', color='#cd5c5c', fillstyle='none', label='SN-GRB (wAGN)', mew=1.5, ms=10)

#print ("AGN", PTFs['PTFname_and_region'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] != "SNsite")].tolist())
ll5 = 'SN-GRB (wAGN)'

l6 = lower_left.errorbar(PTFs['logNII_Ha'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] == "SNsite")], PTFs['logOIII_Hb'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] == "SNsite")], \
xerr=PTFs['elogNII_Ha'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] == "SNsite")], yerr=PTFs['elogOIII_Hb'][(PTFs['SNtype'] == 'GRB') & (PTFs['region'] == "SNsite")], \
fmt='D', color = '#cd5c5c', label='SN-GRB', mew=1.5, ms=10, markeredgecolor='none')

ll6 = 'SN-GRB'

l7 = lower_left.errorbar(PTFs['logNII_Ha'][PTFs['SNtype'] == 'uncertain'], PTFs['logOIII_Hb'][PTFs['SNtype'] == 'uncertain'], \
xerr=PTFs['elogNII_Ha'][PTFs['SNtype'] == 'uncertain'], yerr=PTFs['elogOIII_Hb'][PTFs['SNtype'] == 'uncertain'], \
fmt='g^', fillstyle='none', label='weird/uncertain SN subtype', mew=1.5, ms=10)

ll7 = 'weird/uncertain SN subtype'

# mark upper limit -- again, don't need this anymore because we are removing 2009nz
'''x3 = float(PTFs.loc[PTFs['PTFname_and_region'] == 'GRB091127/SN2009nz-host']['logNII_Ha'])
y3 = float(PTFs.loc[PTFs['PTFname_and_region'] == 'GRB091127/SN2009nz-host']['logOIII_Hb'])
print('GRB091127/SN2009nz-host: logNII_Ha, logOIII_Hb: ', x3, y3)
lower_left.arrow(x3, y3, -0.25, 0, color='red', head_width=0.04)'''

# test for BPT -- don't need this anymore because we are removing 2009nz
'''x4 = float(np.log10(PTFs.loc[PTFs['PTFname_and_region'] == 'GRB091127/SN2009nz-host']['f_SII0']/PTFs.loc[PTFs['PTFname_and_region'] == 'GRB091127/SN2009nz-host']['f_Ha']))
y4 = float(np.log10(PTFs.loc[PTFs['PTFname_and_region'] == 'GRB091127/SN2009nz-host']['f_OIII']/
PTFs.loc[PTFs['PTFname_and_region'] == 'GRB091127/SN2009nz-host']['f_Hb']))
print('GRB091127/SN2009nz-host: logSII_Ha, logOIII_Hb: ', x4, y4)'''

if DOTESTS:
        d, pvalue = stats.ks_2samp(PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic-BL'].tolist(), PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic'].tolist())
        print('*Ic vs. Ic-BL logNII_Ha KS test P-value: ', pvalue)
        print('sizes of the two samples: ', len(PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic'].tolist()), len(PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic-BL'].tolist()))


        d, pvalue = stats.ks_2samp(PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic-BL'].tolist(), PTFs['logNII_Ha'][PTFs['SNtype'] == 'GRB'].tolist())

        print('*Ic-BL vs. SN-GRB logNII_Ha KS test P-value: ', pvalue)
        print('sizes of the two samples: ', len(PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic-BL'].tolist()), len(PTFs['logNII_Ha'][PTFs['SNtype'] == 'GRB'].tolist()))
        #print 'Ic-BL: ', PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic-BL'].tolist()
        #print 'SN-GRB: ', PTFs['logNII_Ha'][PTFs['SNtype'] == 'GRB'].tolist()
        #print 'names: ', PTFs['PTFname_and_region'][PTFs['SNtype'] == 'GRB'].tolist()

        temp1 = PTFs['logOIII_Hb'][PTFs['SNtype'] == 'Ic'].tolist()
        temp2 = PTFs['logOIII_Hb'][PTFs['SNtype'] == 'Ic-BL'].tolist()
        data1 = [x for x in temp1 if x > -10]
        data2 = [x for x in temp2 if x > -10]
        d, pvalue = stats.ks_2samp(data1, data2)
        print('*Ic vs. Ic-BL logOIII_Hb KS test P-value', pvalue)
        print('sizes of the two samples: ', len(data1), len(data2))
        #print 'Ic: ', data1
        #print 'Ic-BL: ', data2

        
lx = np.arange(-2.5, 2.5, 0.001)
print (lx)
#top_histogram.hist(PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic'].tolist(), bins=1000, normed=1, cumulative=True, color='black', histtype='stepfilled')

for sntype in ['Ic-BL', 'Ic', 'GRB', 'sdss', 'slsn']:
        for line,h in zip(['logNII_Ha', 'logOIII_Hb'], [top_histogram, side_histogram]):

            select = PTFs['SNtype'] == sntype
            nx, xbins = np.histogram(PTFs[line][select][np.isfinite(PTFs[line][select])], bins=lx, normed=1, density=True)#, normed=1)#, density=True)#, cumulative=True)
            xtemp1 = [xbins[0]]
            xtemp1.extend(xbins[:-1][~np.isnan(nx)])
            ytemp1 = [0]
            ytemp1.extend(nx[~np.isnan(nx)].cumsum().astype(float)/nx.sum())
            
            nx, xbins = np.histogram((PTFs[line]-PTFs['e'+line])[select][np.isfinite(PTFs[line][select])],
                                      bins=lx, normed=1, density=True)
            xtemp1a = [xbins[0]]
            xtemp1a.extend(xbins[:-1][~np.isnan(nx)])
            ytemp1a = [0]
            ytemp1a.extend(nx[~np.isnan(nx)].cumsum().astype(float)/nx.sum())
            
            nx, xbins = np.histogram((PTFs[line]+PTFs['e'+line])[select][np.isfinite(PTFs[line][select])],
                                     bins=lx, normed=1, density=True)

            #plt.figure()
            #plt.hist((PTFs[line]+PTFs['e'+line])[select][~np.isnan(PTFs[line][select]) * np.isfinite(PTFs[line][select])], bins=lx,
            #                         normed=1, cumulative=True)
            #plt.show()
            xtemp1b = [xbins[0]]
            xtemp1b.extend(xbins[:-1][~np.isnan(nx)])
            ytemp1b = [0]
            ytemp1b.extend(nx[~np.isnan(nx)].cumsum().astype(float)/nx.sum())


            if h == top_histogram:
                    h.plot(xtemp1, ytemp1, color=sncolors[sntype][0], linestyle=sncolors[sntype][1], linewidth=2.5)
                    h.fill_between(xtemp1a, ytemp1a, ytemp1b, color=sncolors[sntype][0], alpha=0.4)
            else:
                    h.plot(ytemp1, xtemp1, color=sncolors[sntype][0], linestyle=sncolors[sntype][1], linewidth=2.5)
                    h.fill_betweenx(xtemp1, ytemp1a, ytemp1b, color=sncolors[sntype][0], alpha=0.4)

                    
#-----------------------------Add SDSS to histograms-----------------------------------------------------------------------------

for line,h in zip(['logNII_Ha', 'logOIII_Hb'], (top_histogram, side_histogram)):

        nx, xbins = np.histogram(sdss[line], bins=lx, normed=1, density=True)#, cumulative=True)
        xtemp4 = [xbins[0]]
        xtemp4.extend(xbins[:-1])
        ytemp4 = [0]
        ytemp4.extend(nx.cumsum().astype(float)/nx.sum())

        nx, xbins= np.histogram((sdss[line]-sdss['e'+line]), bins=lx, normed=1, density=True)#, cumulative=True)
        xtemp4a = [xbins[0]]
        xtemp4a.extend(xbins[:-1])
        ytemp4a = [0]
        ytemp4a.extend(nx.cumsum().astype(float)/nx.sum())

        nx, xbins= np.histogram((sdss[line]+sdss['e'+line]), bins=lx, normed=1, density=True)#, cumulative=True)
        xtemp4b = [xbins[0]]
        xtemp4b.extend(xbins[:-1])
        ytemp4b = [0]
        ytemp4b.extend(nx.cumsum().astype(float)/nx.sum())

        if h==top_histogram:
                h.plot(xtemp4, ytemp4, color=sncolors['sdss'][0], linestyle=sncolors['sdss'][1], linewidth=2.5)
                h.fill_between(xtemp4a, ytemp4a, ytemp4b, color=sncolors['sdss'][0], alpha=0.4)
        else:
                h.plot(ytemp4, xtemp4, color=sncolors['sdss'][0], linestyle=sncolors['sdss'][1], linewidth=2.5)
                h.fill_betweenx(xtemp4, ytemp4a, ytemp4b, color=sncolors['sdss'][0], alpha=0.4)                

                    
#-----------------------------Add SDSS to histograms-----------------------------------------------------------------------------

#-----------------------------Add SLSN to histograms-----------------------------------------------------------------------------


for line,h in zip(['logNII_Ha', 'logOIII_Hb'], (top_histogram, side_histogram)):
        
        if noflags == 'False':
            nx, xbins = np.histogram(slsn[line], bins=lx, normed=1, density=True)#, cumulative=True)
            xtemp4 = [xbins[0]]
            xtemp4.extend(xbins[:-1])
            ytemp4 = [0]
            ytemp4.extend(nx.cumsum().astype(float)/nx.sum())
    
            nx, xbins= np.histogram((slsn[line]-slsn['e'+line]), bins=lx, normed=1, density=True)#, cumulative=True)
            xtemp4a = [xbins[0]]
            xtemp4a.extend(xbins[:-1])
            ytemp4a = [0]
            ytemp4a.extend(nx.cumsum().astype(float)/nx.sum())
    
            nx, xbins= np.histogram((slsn[line]+slsn['e'+line]), bins=lx, normed=1, density=True)#, cumulative=True)
            xtemp4b = [xbins[0]]
            xtemp4b.extend(xbins[:-1])
            ytemp4b = [0]
            ytemp4b.extend(nx.cumsum().astype(float)/nx.sum())
        
        if noflags == 'True':
            #below plot only the slsne without flags
            nx, xbins = np.histogram(slsn[line][ind_noflags], bins=lx, normed=1, density=True)#, cumulative=True)
            xtemp4 = [xbins[0]]
            xtemp4.extend(xbins[:-1])
            ytemp4 = [0]
            ytemp4.extend(nx.cumsum().astype(float)/nx.sum())
    
            nx, xbins= np.histogram((slsn[line][ind_noflags]-slsn['e'+line][ind_noflags]), bins=lx, normed=1, density=True)#, cumulative=True)
            xtemp4a = [xbins[0]]
            xtemp4a.extend(xbins[:-1])
            ytemp4a = [0]
            ytemp4a.extend(nx.cumsum().astype(float)/nx.sum())
    
            nx, xbins= np.histogram((slsn[line][ind_noflags]+slsn['e'+line][ind_noflags]), bins=lx, normed=1, density=True)#, cumulative=True)
            xtemp4b = [xbins[0]]
            xtemp4b.extend(xbins[:-1])
            ytemp4b = [0]
            ytemp4b.extend(nx.cumsum().astype(float)/nx.sum())

        if h==top_histogram:
                h.plot(xtemp4, ytemp4, color=sncolors['slsn'][0], linestyle=sncolors['slsn'][1], linewidth=2.5)
                h.fill_between(xtemp4a, ytemp4a, ytemp4b, color=sncolors['slsn'][0], alpha=0.4)
        else:
                h.plot(ytemp4, xtemp4, color=sncolors['slsn'][0], linestyle=sncolors['slsn'][1], linewidth=2.5)
                h.fill_betweenx(xtemp4, ytemp4a, ytemp4b, color=sncolors['slsn'][0], alpha=0.4)                

#-----------------------------Add SLSN to histograms-----------------------------------------------------------------------------


##########labels. arrows, fine tuning
plt.annotate('HII', xy=(0.2, 0.85), xycoords='axes fraction', fontsize=fs-3, color='grey')
plt.annotate('AGN', xy=(0.8, 0.85), xycoords='axes fraction', fontsize=fs-3, color='grey')

#lower_left.text(-1., 0.21, "PTF09sk", ha="center", va="center", color="#386890")
lower_left.text(-1.35, 0.9, "GRB031203/SN2003lw", ha="left", va="center", color="IndianRed", fontsize=fs-5)



x1 = float(PTFs.loc[PTFs['PTFname_and_region'] == 'GRB130702A/SN2013dx-host', 'logNII_Ha'])
y01 = float(PTFs.loc[PTFs['PTFname_and_region'] == 'GRB130702A/SN2013dx-host', 'logOIII_Hb'])
# this value is set by hand
y1 = 3.5*(1/float(no_GRBs))

print ("here", x1, y01)
lower_left.arrow(x1 - 0.05, y01, -0.25, 0, color='IndianRed', head_width=0.04)

top_histogram.arrow(x1, y1, -0.25, 0, color='IndianRed', head_width=0.04)
top_histogram.text(x1-0.01, y1 + 0.05, "GRB130702A/SN2013dx",
				 ha="right", color="IndianRed", fontsize=fs-5)

##remove this, because we don't need 2009nz anymore
'''x2 = float(PTFs.loc[PTFs['PTFname_and_region'] == 'GRB091127/SN2009nz-host', 'logNII_Ha'])
y2 = 10.5*(1/float(no_GRBs))
top_histogram.arrow(x2, y2, -0.25, 0, color='red', head_width=0.04)'''

top_histogram.set_xticklabels([])
top_histogram.set_yticklabels([0.5, 1])
top_histogram.set_yticks([0.5, 1])
plt.setp(top_histogram.get_yticklabels(), fontsize=fs-5)
top_histogram.set_ylim(0, 1)

side_histogram.set_yticklabels([])
side_histogram.set_xticklabels([0.5, 1])
side_histogram.set_xticks([0.5, 1])
plt.setp(side_histogram.get_xticklabels(), fontsize=fs-5)
side_histogram.set_xlim(0, 1)

if DOTESTS:
        d, pvalue = stats.ks_2samp(data1, data2)
        print('*Ic-BL vs. SN-GRB logOIII_Hb KS test P-value: ', pvalue)
        print('sizes of the two samples: ', len(data1), len(data2))

        #data2 = [x for x in data2 if x < 0.8]
        #d, pvalue = stats.ks_2samp(data1, data2)
        #print '*Ic-BL vs. SN-GRB (excluding AGN) logOIII_Hb KS test P-value: ', pvalue
        #print 'sizes of the two samples: ', len(data1), len(data2)
        print('** Need to do test on logOIII_Hb for Ic-BL vs. SN-GRB wo AGN')
        
        print ("   ")



#print PTFs[['f_NII', 'f_Ha', 'logNII_Ha', 'logOIII_Hb', 'PTFname_and_region', 'region']][(PTFs['SNtype'] == 'GRB')]

'''legend = lower_left.legend(loc='lower left', fontsize=fs-8, frameon=False, numpoints=1)
for lh in legend.legendHandles:
	lh.set_alpha(1)'''

sdsspoint, = plt.plot([-3.,], [-3.], marker='o', color='#FFD700', ms=10, mew=1.5)
sdss_label = 'SDSS galaxies'

plt.legend((sdsspoint, l1, l2, l3,l4,l5,l6,l7, l_slsn),(sdss_label, ll1,ll2,ll3,ll4,ll5,ll6,ll7, label_slsn),
           loc='lower left', fontsize=fs-8, frameon=False, numpoints=1)

for ax in [top_histogram, lower_left]:
	ax.set_xlim(x_range[0], x_range[1])
for ax in [side_histogram, lower_left]:
	ax.set_ylim(y_range[0], y_range[1])


for ax in [lower_left, top_histogram, side_histogram]:
	ax.minorticks_on()
	plt.setp(ax.yaxis.get_ticklines(), 'markersize', 10)
	plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 1)
	plt.setp(ax.xaxis.get_ticklines(), 'markersize', 10)
	plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 1)
	plt.setp(ax.yaxis.get_ticklines(minor=True), 'markersize', 5)
	plt.setp(ax.yaxis.get_ticklines(minor=True), 'markeredgewidth', 1)
	plt.setp(ax.xaxis.get_ticklines(minor=True), 'markersize', 5)
	plt.setp(ax.xaxis.get_ticklines(minor=True), 'markeredgewidth', 1)

if noflags == 'False':
    plt.savefig('BPT_042318.pdf', format='pdf')
    plt.close(fig)
if noflags == 'True':
    plt.savefig('BPT_042318_noflags.pdf', format='pdf')
    plt.close(fig)




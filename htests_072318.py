'''
apply hypothesis tests to see if the two samples are drawn from the same underlying distributions
'''

import numpy as np
import pandas as pd
from scipy import stats
from astropy.io import ascii


def htests(data1, data2):
    d, pvalue = stats.ks_2samp(data1, data2)
    print(' KS test:        ', pvalue)
    statistic, criticalv, significance = stats.anderson_ksamp([data1, data2])
    print(' AD test:        ', significance)
    statistic, pvalue = stats.ranksums(data1, data2)
    print(' Wilcoxon test:  ', pvalue)
    return

def datasum(data):
    #print '        mean    error   std     25th    median  75th    N'
    x1 = np.mean(data)
    x2 = stats.sem(data)
    x3 = np.std(data)
    x4 = np.percentile(data, 25)
    x5 = np.percentile(data, 50)
    x6 = np.percentile(data, 75)
    x7 = len(data)
    #x8 = np.median(data) # same as x5
    #x8 = x3/np.sqrt(x7-1) # same as x2
    #print '%13.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8d' % (x1, x2, x3, x4, x5, x6, x7)
    return x7, x1, x2, x4, x5, x6


def bpttests(ptffluxfile, grbfluxfile):

    names = ['name', \
    'f_OII', 'ef_OII', 'f_Hb', 'ef_Hb', 'f_OIII0', 'ef_OIII0', 'f_OIII', 'ef_OIII', \
    'f_Ha', 'ef_Ha', 'f_NII', 'ef_NII', 'f_SII0', 'ef_SII0', 'f_SII1', 'ef_SII1']

    PTFs = pd.read_csv(ptffluxfile, skiprows=1, delim_whitespace=True, header=None, na_values=['...'], names=names)
    PTFs['SNtype'] = 'PTF'
    # modified for the 072117 version that place the uncertain types at the beginning
    #PTFs.loc[0:4, 'SNtype'] = 'uncertain'
    PTFs.loc[0:5, 'SNtype'] = 'uncertain'
    #PTFs.loc[5:18, 'SNtype'] = 'Ic-BL'
    PTFs.loc[6:19, 'SNtype'] = 'Ic-BL'
    #PTFs.loc[19:, 'SNtype'] = 'Ic'
    PTFs.loc[20:, 'SNtype'] = 'Ic'

    GRBs = pd.read_csv(grbfluxfile, skiprows=1, delim_whitespace=True, header=None, na_values=['...'], names=names)
    GRBs['SNtype'] = 'GRB'

    #print GRBs, len(GRBs)
    #raw_input()
    GRBs = GRBs[0:11]
    GRBs = GRBs[~(GRBs.name == 'GRB091127/SN2009nz')]
    #print GRBs, len(GRBs)
    #raw_input()

    PTFs = pd.concat([PTFs, GRBs], ignore_index=True)
    #print len(PTFs)
    #raw_input()
    PTFs['logOIII_Hb'] = np.log10(PTFs['f_OIII']/PTFs['f_Hb'])
    PTFs['logNII_Ha'] = np.log10(PTFs['f_NII']/PTFs['f_Ha'])

    print('1. logNII_Ha: ')
    data1 = PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic'].tolist()
    data2 = PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic-BL'].tolist()
    print('*Ic vs. Ic-BL (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    data1 = PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic-BL'].tolist()
    data2 = PTFs['logNII_Ha'][PTFs['SNtype'] == 'GRB'].tolist()
    print('*Ic-BL vs. SN-GRB (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)
    #sys.exit()
    #print 'Ic-BL: ', PTFs['logNII_Ha'][PTFs['SNtype'] == 'Ic-BL'].tolist()
    #print 'SN-GRB: ', PTFs['logNII_Ha'][PTFs['SNtype'] == 'GRB'].tolist()
    #print 'names: ', PTFs['name'][PTFs['SNtype'] == 'GRB'].tolist()

    #data2 = PTFs['logNII_Ha'][(PTFs['SNtype'] == 'GRB') & (PTFs['name'] != 'GRB031203/SN2003lw')].tolist()
    data2 = PTFs['logNII_Ha'][(PTFs['SNtype'] == 'GRB') & (PTFs['name'] != 'GRB031203/SN2003lw') & (PTFs['name'] != 'GRB091127/SN2009nz')].tolist()
    print('*Ic-BL vs. SN-GRB excluding AGN (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    print('2. logOIII_Hb: ')
    data1 = PTFs['logOIII_Hb'][(PTFs['SNtype'] == 'Ic') & (PTFs['logOIII_Hb'] > -10)].tolist()
    data2 = PTFs['logOIII_Hb'][PTFs['SNtype'] == 'Ic-BL'].tolist()
    print('*Ic vs. Ic-BL (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    data1 = PTFs['logOIII_Hb'][PTFs['SNtype'] == 'Ic-BL'].tolist()
    data2 = PTFs['logOIII_Hb'][(PTFs['SNtype'] == 'GRB') & (PTFs['logOIII_Hb'] > -10)].tolist()
    print('*Ic-BL vs. SN-GRB (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    data2 = PTFs['logOIII_Hb'][(PTFs['SNtype'] == 'GRB') & (PTFs['name'] != 'GRB031203/SN2003lw') & (PTFs['name'] != 'GRB091127/SN2009nz') & (PTFs['logOIII_Hb'] > -10)].tolist()
    print('*Ic-BL vs. SN-GRB excluding AGN (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)


#def ohtests(ptfohfile, grbohfile, slsnohfile, outfile, OHfile1, OHfile2):
    
ptffluxfile = 'ptfflux_072117.txt'
grbfluxfile = 'GRBflux_083017.txt'
slsnohfile='SLSN_i_z.txt'
ptfohfile = 'ptfoh_072617.txt'
grbohfile = 'grboh_083017.txt'
outfile = 'table7_072318.tex'
OHfile1 = 'platefitflux_072617.pkl'
OHfile2 = 'GRB_083017.pkl'

fout = open(outfile, 'w')
lines = r'''
\begin{deluxetable}{lcccccc}
\tabletypesize{\small}
\tablenum{7}
\tablewidth{0pt}
\tablecaption{Summary of the distributions of metallicities on various scales}
\tablehead{
\colhead{SN type} & \colhead{$N$} & \colhead{Mean} & \colhead{SEM} & \colhead{25th} & \colhead{Median} & \colhead{75th}
}
\startdata
'''
fout.write(lines)

'''
Below are the following assumptions:
    
Perley calibration name vs. our calibration name
KD02 = KD02comb
PP04O3 = PP04_O3N2
M08 = M08_N2Ha
M13O3 = M13_O3N2
'''

OHscales = ['E(B-V)', 'KD02comb', 'PP04_O3N2', 'M08_N2Ha', 'M13_O3N2']
OHscalenames = [r'KD02\_COMB', r'PP04\_O3N2', r'M08\_N2H$\alpha$', r'M13\_O3N2']
names = ['name', 'D13_N2S2_O3S2', 'Dm', 'Dp', 'KD02comb', 'Km', 'Kp', 'PP04_O3N2', 'Pm', 'Pp', 'M08_N2Ha', 'Mm', 'Mp', 'E(B-V)', 'Em', 'Ep', 'M13_O3N2', 'M13m', 'M13p']
'''
OHscales = ['E(B-V)', 'D13_N2S2_O3S2', 'KD02comb', 'PP04_O3N2', 'M08_N2Ha']
OHscalenames = [r'D13\_N2S2\_O3S2', r'KD02\_COMB', r'PP04\_O3N2', r'M08\_N2H$\alpha$']
names = ['name', 'D13_N2S2_O3S2', 'Dm', 'Dp', 'KD02comb', 'Km', 'Kp', 'PP04_O3N2', 'Pm', 'Pp', 'M08_N2Ha', 'Mm', 'Mp', 'E(B-V)', 'Em', 'Ep', 'M13_O3N2', 'M13m', 'M13p']
'''
PTFs = pd.read_csv(ptfohfile, skiprows=1, delim_whitespace=True, header=None, na_values=['...'], names=names)
PTFs['SNtype'] = 'PTF'
#PTFs.loc[0:4, 'SNtype'] = 'uncertain'
PTFs.loc[0:5, 'SNtype'] = 'uncertain'
#PTFs.loc[5:18, 'SNtype'] = 'Ic-BL'
PTFs.loc[6:19, 'SNtype'] = 'Ic-BL'
#PTFs.loc[19:, 'SNtype'] = 'Ic'
PTFs.loc[20:, 'SNtype'] = 'Ic'

GRBs = pd.read_csv(grbohfile, skiprows=1, delim_whitespace=True, header=None, na_values=['...'], names=names)
GRBs['SNtype'] = 'GRB'
GRBs = GRBs[~(GRBs.name == 'GRB091127/SN2009nz')]

PTFs = pd.concat([PTFs, GRBs], ignore_index=True)

SLSNe = ascii.read(slsnohfile)

for i, OHscale in enumerate(OHscales):
    print('%1d. %s' % (i, OHscale))

    data1 = PTFs[OHscale][(PTFs['SNtype'] == 'Ic') & (PTFs[OHscale] > -10)].tolist()
    data2 = PTFs[OHscale][(PTFs['SNtype'] == 'Ic-BL') & (PTFs[OHscale] > -10)].tolist()
    print('*Ic vs. Ic-BL (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    #print '*Ic: mean       error   std     25th    median  75th    N'
    x7, x1, x2, x4, x5, x6 = datasum(data1)
    if i>0:
        line = r'&\multicolumn{6}{c}{'+OHscalenames[i-1]+r'}\\' + '\n'
        fout.write(line)
        line = r'\hline' + '\n'
        fout.write(line)
        line = r'PTF SN Ic & '+str(int(x7))+r'&'+str(round(x1, 3))+r'&'+str(round(x2, 3))+r'&'+str(round(x4, 3))+r'&'+str(round(x5, 3))+r'&'+str(round(x6, 3))+r'\\'+'\n'
        fout.write(line)
    #print '*Ic-BL: mean     error   std     25th    median  75th    N'
    x7, x1, x2, x4, x5, x6 = datasum(data2)
    if i>0:
        line = r'PTF SN Ic-BL & '+str(int(x7))+r'&'+str(round(x1, 3))+r'&'+str(round(x2, 3))+r'&'+str(round(x4, 3))+r'&'+str(round(x5, 3))+r'&'+str(round(x6, 3))+r'\\'+'\n'
        fout.write(line)

    data_icbl = PTFs[OHscale][(PTFs['SNtype'] == 'Ic-BL') & (PTFs[OHscale] > -10)].tolist()
    data_grb_with_agn = PTFs[OHscale][(PTFs['SNtype'] == 'GRB') & (PTFs[OHscale] > -10)].tolist()
    print('*Ic-BL vs. SN-GRB (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data_icbl, data_grb_with_agn)

    #print '*GRB:   mean    error   std     25th    median  75th    N'
    x7, x1, x2, x4, x5, x6 = datasum(data_grb_with_agn)
    if i>1: # SN2003lw and SN2009nz have no D13 scale
        line = r'SN-GRB (including AGNs) & '+str(int(x7))+r'&'+str(round(x1, 3))+r'&'+str(round(x2, 3))+r'&'+str(round(x4, 3))+r'&'+str(round(x5, 3))+r'&'+str(round(x6, 3))+r'\\'+'\n'
        fout.write(line)

    data_grb_no_agn = PTFs[OHscale][(PTFs['SNtype'] == 'GRB') & (PTFs[OHscale] > -10) & (PTFs['name'] != 'GRB031203/SN2003lw') & (PTFs['name'] != 'GRB091127/SN2009nz')].tolist()
    print('*Ic-BL vs. SN-GRB excluding AGN (%2d vs. %2d)' %  (len(data_icbl), len(data_grb_no_agn)))
    htests(data_icbl, data_grb_no_agn)

    #print '*GRBwoAGN: mean error   std     25th    median  75th    N'
    x7, x1, x2, x4, x5, x6 = datasum(data_grb_no_agn)
    if i>0:
        line = r'SN-GRB (excluding AGNs) & '+str(int(x7))+r'&'+str(round(x1, 3))+r'&'+str(round(x2, 3))+r'&'+str(round(x4, 3))+r'&'+str(round(x5, 3))+r'&'+str(round(x6, 3))+r'\\'+'\n'
        fout.write(line)
        
    if i>0:
        data_slsn = SLSNe[OHscale][np.where(SLSNe[OHscale] != 0.0)]
        print('*Ic-BL vs. SLSNe (%2d vs. %2d)' %(len(data_icbl), len(data_slsn)))
        htests(data_icbl, data_slsn)
        x7, x1, x2, x4, x5, x6 = datasum(data_slsn)
        
        line = r'SLSNe & ' +str(int(x7))+r'&'+str(round(x1, 3))+r'&'+str(round(x2, 3))+r'&'+str(round(x4, 3))+r'&'+str(round(x5, 3))+r'&'+str(round(x6, 3))+r'\\'+'\n'
        fout.write(line)
        
        print('*SN-GRB (without AGN) vs. SLSNe (%2d vs. %2d)' %(len(data_grb_no_agn), len(data_slsn)))
        htests(data_grb_no_agn, data_slsn)
    
        print('*SN-GRB (with AGN) vs. SLSNe (%2d vs. %2d)' %(len(data_grb_with_agn), len(data_slsn)))
        htests(data_grb_with_agn, data_slsn)

    if i<4 and i>0:
        line = r'\hline'
        fout.write(line)

lines = r'''\enddata
\end{deluxetable}
        '''
fout.write(lines)
fout.close()

'''

#data1 = PTFs[''][(PTFs['SNtype'] == 'Ic-BL') & (PTFs[OHscale] > -10)].tolist()
#print(PTFs[['PP04_O3N2', 'SNtype']])
data = PTFs['PP04_O3N2'][(PTFs['SNtype'] != 'uncertain') & (PTFs['PP04_O3N2'] > -10)].tolist()
print('Combined:' )
print('1. PP04_O3N2: mean error   std     25th    median  75th    N')
x7, x1, x2, x4, x5, x6 = datasum(data)
print('%13.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8d' % (x1, x2, x2*np.sqrt(x7-1), x4, x5, x6, x7))

PTFs = pd.read_pickle(OHfile1)
GRBs = pd.read_pickle(OHfile2)

GRBs = np.array([GRBs[i] for i in range(len(GRBs)) if not GRBs[i]['PTFname']== 'SN2009nz'])

data1 = [x['PP04_O3N2'][0] for x in PTFs if ((x['SNtype'] != 'uncertain') and (x['SNtype'] != 'other') and ('PP04_O3N2' in x.keys()))]
data2 = [x['PP04_O3N2'][0] for x in GRBs if ((x['SNtype'] == 'GRB') and ('PP04_O3N2' in x.keys()))]
data = np.array(data1 + data2)
x7, x1, x2, x4, x5, x6 = datasum(data)
print('%13.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8d' % (x1, x2, x2*np.sqrt(x7-1), x4, x5, x6, x7))

print('2. M13_O3N2: mean error   std     25th    median  75th    N')
data1 = [x['M13_O3N2'][0] for x in PTFs if ((x['SNtype'] != 'uncertain') and (x['SNtype'] != 'other') and ('M13_O3N2' in x.keys()))]
data2 = [x['M13_O3N2'][0] for x in GRBs if ((x['SNtype'] == 'GRB') and ('M13_O3N2' in x.keys()))]
data = np.array(data1 + data2)
x7, x1, x2, x4, x5, x6 = datasum(data)
print('%13.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8d' % (x1, x2, x2*np.sqrt(x7-1), x4, x5, x6, x7))
'''
'''
def sedtests(ptfsedfile, grbsedfile):

    names = ['name', 'u', 'ue', 'g', 'ge', 'r', 're', 'i', 'ie', 'z', 'ze', 'FUV', 'FUVe', 'NUV', 'NUVe', 'logMs', 'logMsm', 'logMsp', 'logSFR', 'logSFRm', 'logSFRp']
    PTFs = pd.read_csv(ptfsedfile, skiprows=1, delim_whitespace=True, header=None, na_values=['...'], names=names)
    PTFs['SNtype'] = 'PTF'
    PTFs.loc[0:13, 'SNtype'] = 'Ic-BL'
    #PTFs.loc[14:43, 'SNtype'] = 'Ic'
    PTFs.loc[14:41, 'SNtype'] = 'Ic'
    #PTFs.loc[44:, 'SNtype'] = 'uncertain'
    PTFs.loc[42:, 'SNtype'] = 'uncertain'
    PTFs['badsfr'] = PTFs.apply(lambda row: 1 if np.isnan(row['FUV']) else 0, axis = 1)

    GRBs = pd.read_csv(grbsedfile, skiprows=2, delim_whitespace=True, header=None, index_col='galnum', \
    names=['galnum', 'reference', 'hostname_and_region', 'logMs', 'elogMs', 'SFR', 'eSFR', 'logMs_L', 'elogMs_L'], na_values='-')
    #GRBs = GRBs[0:9]
    #GRBs = GRBs[0:10]
    GRBs = GRBs[0:11]
    GRBs['SNtype'] = 'GRB'

    GRBs['hostname'] = GRBs['hostname_and_region'].str.split('-').str[0].str[-8:]
    GRBs.loc['3']['hostname'] = 'XRF020903'
    GRBs.loc['10']['hostname'] = 'SN2016jca'
    #print GRBs
    #raw_input()

    GRBs = GRBs[~(GRBs.hostname == 'SN2009nz')]
    #print GRBs
    #raw_input()
    #print GRBs

    GRBs['logMs'] = GRBs['logMs'].astype(np.float)
    GRBs['logSFR'] = np.log10(np.array(GRBs['SFR']).astype(np.float))
    GRBs['elogSFR'] = np.array(GRBs['eSFR']).astype(np.float)/(np.array(GRBs['SFR']).astype(np.float)*np.log(10.))



    print('1. logMs')

    data1 = PTFs['logMs'][PTFs['SNtype'] == 'Ic'].tolist()
    data2 = PTFs['logMs'][PTFs['SNtype'] == 'Ic-BL'].tolist()
    print('*Ic vs. Ic-BL (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    data1 = PTFs['logMs'][PTFs['SNtype'] == 'Ic-BL'].tolist()
    data2 = GRBs['logMs'].tolist()
    print('*Ic-BL vs. SN-GRB (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    data2 = GRBs['logMs'][(GRBs['hostname'] != 'SN2003lw') & (GRBs['hostname'] != 'SN2009nz')].tolist()
    print('*Ic-BL vs. SN-GRB excluding AGN (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    print('2. logSFR')

    data1 = PTFs['logSFR'][PTFs['SNtype'] == 'Ic'].tolist()
    data2 = PTFs['logSFR'][PTFs['SNtype'] == 'Ic-BL'].tolist()
    print('*Ic vs. Ic-BL (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    data1 = PTFs['logSFR'][PTFs['SNtype'] == 'Ic-BL'].tolist()
    data2 = GRBs['logSFR'].tolist()
    print('*Ic-BL vs. SN-GRB (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    data1 = PTFs['logSFR'][(PTFs['SNtype'] == 'Ic-BL') & (PTFs['badsfr'] == 0)].tolist()
    data2 = GRBs['logSFR'][(GRBs['hostname'] != 'SN2003lw') & (GRBs['hostname'] != 'SN2009nz')].tolist()
    print('*Ic-BL (good SFR) vs. SN-GRB excluding AGN (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    print('3. logsSFR')
    data1 = (PTFs['logSFR'][PTFs['SNtype'] == 'Ic']-PTFs['logMs'][PTFs['SNtype'] == 'Ic']).tolist()
    data2 = (PTFs['logSFR'][PTFs['SNtype'] == 'Ic-BL']-PTFs['logMs'][PTFs['SNtype'] == 'Ic-BL']).tolist()
    print('*Ic vs. Ic-BL (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    data1 = (PTFs['logSFR'][PTFs['SNtype'] == 'Ic-BL']-PTFs['logMs'][PTFs['SNtype'] == 'Ic-BL']).tolist()
    data2 = (GRBs['logSFR']-GRBs['logMs']).tolist()
    print('*Ic-BL vs. SN-GRB (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    data1 = (PTFs['logSFR'][(PTFs['SNtype'] == 'Ic-BL') & (PTFs['badsfr'] == 0)]-PTFs['logMs'][(PTFs['SNtype'] == 'Ic-BL') & (PTFs['badsfr'] == 0)]).tolist()
    data2 = (GRBs['logSFR'][(GRBs['hostname'] != 'SN2003lw') & (GRBs['hostname'] != 'SN2009nz')] - GRBs['logMs'][(GRBs['hostname'] != 'SN2003lw') & (GRBs['hostname'] != 'SN2009nz')]).tolist()
    print('*Ic-BL (good SFR) vs. SN-GRB excluding AGN (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

    data2 = (PTFs['logSFR'][(PTFs['SNtype'] == 'Ic') & (PTFs['badsfr'] == 0)]-PTFs['logMs'][(PTFs['SNtype'] == 'Ic') & (PTFs['badsfr'] == 0)]).tolist()
    print('*Ic-BL (good SFR) vs. Ic (good SFR) (%2d vs. %2d)' %  (len(data1), len(data2)))
    htests(data1, data2)

def main():
    #ptffluxfile = '/home/sh162/work/platefit/platefit/4pyMCZ/python/ptfflux_072117.txt'
    ptffluxfile = 'ptfflux_072117.txt'
    #grbfluxfile = '/home/sh162/work/platefit/platefit/4pyMCZ/python/grbflux_042117.txt'
    #grbfluxfile = '/home/sh162/work/platefit/platefit/4pyMCZ/python/grbflux_083017.txt'
    #grbfluxfile = 'grbflux_083017.txt'
    grbfluxfile = 'GRBflux_083017.txt'
    bpttests(ptffluxfile, grbfluxfile)
    slsnohfile='SLSN_z_i.txt'
    print('_______________')
    #ptfohfile = 'ptfoh_072117.txt'
    ptfohfile = 'ptfoh_072617.txt'
    #grbohfile = 'grboh_042117.txt'
    grbohfile = 'grboh_083017.txt'
    #outfile = 'table7_072617.tex'
    outfile = 'table7_062818.tex'
    #OHfile1 = '/home/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_072617.pkl'
    OHfile1 = 'platefitflux_072617.pkl'
    #OHfile2 = '/home/sh162/work/platefit/platefit/4pyMCZ/python/GRB_042117.pkl'
    #OHfile2 = '/home/sh162/work/platefit/platefit/4pyMCZ/python/GRB_083017.pkl'
    OHfile2 = 'GRB_083017.pkl'
    ohtests(ptfohfile, grbohfile, slsnohfile, outfile, OHfile1, OHfile2)

    print('_______________')
    #ptfsedfile = '/home/sh162/work/SED/fromDan/ptfsed_072117.txt'
    #ptfsedfile = '/home/sh162/work/SED/fromDan/ptfsed_072617.txt'
    ptfsedfile = 'ptfsed_072617.txt'
    #grbsedfile = '/home/sh162/work/platefit/platefit/4pyMCZ/GRB/GRBMsSFR_030117.txt'
    #grbsedfile = '/home/sh162/work/platefit/platefit/4pyMCZ/GRB/GRBMsSFR_081117.txt'
    #grbsedfile = '/home/sh162/work/platefit/platefit/4pyMCZ/GRB/GRBMsSFR_083017.txt'
    #grbsedfile = '/home/sh162/work/platefit/platefit/4pyMCZ/GRB/GRBMsSFR_083017.txt'
    grbsedfile = 'GRBMsSFR_083017.txt'
    sedtests(ptfsedfile, grbsedfile)


if __name__ == '__main__':
    main()
'''
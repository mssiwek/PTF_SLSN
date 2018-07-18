# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from astropy.io import ascii
import os

fp = os.getcwd()
fname_5 ='/apjaa3522t5_mrt.txt'
fname_6 ='/apjaa3522t6_mrt.txt'
fname_1 ='/apjaa3522t1_mrt.txt'

slsn_t1 = ascii.read(fp+fname_1)
data_5 = ascii.read(fp + fname_5)
data_6 = ascii.read(fp + fname_6)

#find the indeces at which the slsn hosts are type I or I-R, but NOT II:
type_i_slsn = np.where(slsn_t1['Class'] != 'II')
#the trouble is that in table 1 (slsn_t1) there are exactly 32 entries because we 
#have 32 different galaxies. But in other tables there are multiple measurements
#of one galaxy: 10uhf, 10uhf (site) -- 11rks, 11rks (site)
#so need to find the index using the names.
#below use the previously obtained index array to get the galaxy host NAMES from table 1, which are then used to get the index from the tables with duplicates:
type_i_slsn_names = slsn_t1['PTF'][type_i_slsn]

type_i_slsn_index = []
for type_i_slsn_name in type_i_slsn_names:
    aa = np.where(data_5['PTF'] == type_i_slsn_name)
    #print type_i_slsn_name
    #print aa
    #print len(aa)
    #print len(aa[0])
    if len(aa[0])==1:
        type_i_slsn_index.append(aa[0][0])
    else:
        #print aa[0]
        for aaa in aa[0]:
            #print aaa
            type_i_slsn_index.append(aaa)
    
data_5_i_only = data_5[type_i_slsn_index]
#But in table 6 the galaxies are not duplicate, so can use the original index:
data_6_i_only = data_6[type_i_slsn]

keys_5 = ['PTF','flag', 'F[OII]3727', 'e_F[OII]3727', 'FHb', 'e_FHb', 'F[OIII]4959', 'e_F[OIII]4959', 'l_F[OIII]5007','F[OIII]5007', 'e_F[OIII]5007', 'l_FHa','FHa', 'e_FHa',  'l_F[NII]6584','F[NII]6584', 'e_F[NII]6584', 'F[NII]6548', 'e_F[NII]6548','F[SII]6716', 'e_F[SII]6716', 'F[SII]6731', 'e_F[SII]6731']
f_names = ['name','flag','f_OII','ef_OII','f_Hb','ef_Hb','f_OIII_4959','ef_OIII_4959','l_OIII_5007','f_OIII_5007','ef_OIII_5007', 'l_Ha','f_Ha','ef_Ha', 'l_NII_6584','f_NII_6584','ef_NII_6584','f_NII_6548','ef_NII_6548','f_SII_6717','ef_SII_6717','f_SII_6731','ef_SII_6731']

keys_6 = ['PTF', 'KD02', 'e_KD02', 'E_KD02', 'PP04O3', 'e_PP04O3', 'E_PP04O3', 'M08', 'e_M08', 'E_M08', 'M13O3', 'e_M13O3', 'E_M13O3']
z_names = ['name', 'KD02comb', 'ne_KD02comb', 'pe_KD02comb','PP04_O3N2', 'ne_PP04_O3N2','pe_PP04_O3N2', 'M08_N2Ha', 'ne_M08_N2Ha', 'pe_M08_N2Ha','M13_O3N2', 'ne_M13_O3N2', 'pe_M13_O3N2']

data_5_concat = np.concatenate([data_5[keys_5]]) 
data_5_i_only_concat = np.concatenate([data_5_i_only[keys_5]])  
data_6_concat = np.concatenate([data_6[keys_6]]) 
data_6_i_only_concat = np.concatenate([data_6_i_only[keys_6]])  

for g in range(0, len(data_5_i_only_concat)):
    if data_5_i_only_concat['flag'][g] == 'c':
        data_5_i_only_concat['PTF'][g] = data_5_i_only_concat['PTF'][g]+'*'
        print data_5_i_only_concat['PTF'][g]
    if data_5_i_only_concat['flag'][g] == 'd':
        data_5_i_only_concat['PTF'][g] = data_5_i_only_concat['PTF'][g]+'+'
        print data_5_i_only_concat['PTF'][g]
    if data_5_i_only_concat['flag'][g] == 's':
        data_5_i_only_concat['PTF'][g] = data_5_i_only_concat['PTF'][g]+'#'
        print data_5_i_only_concat['PTF'][g]
        
    
ascii.write(data_5_concat, 'slsn_flux.txt', names = f_names, overwrite=True)
ascii.write(data_5_i_only_concat, 'slsn_i_flux.txt', names = f_names, overwrite=True)
#ascii.write(data_5_i_only_concat, 'pyMCZ-master/input/slsn_i_flux.txt', names = f_names, overwrite=True)

ascii.write(data_6_concat, 'slsn_z.txt', names = z_names, overwrite=True)
ascii.write(data_6_i_only_concat, 'slsn_i_z.txt', names = z_names, overwrite=True)
#ascii.write(data_6_i_only_concat, 'pyMCZ-master/output/slsn_i/slsn_z_perley.txt', names = z_names, overwrite=True)



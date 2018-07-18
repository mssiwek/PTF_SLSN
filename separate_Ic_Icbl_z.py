#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:55:22 2018

@author: magda
"""

from astropy.io import ascii
import os
import pandas as pd

filename = '/ptfoh_072617.txt'
fp = os.getcwd()
no_ic = 28
no_icbl = 14
no_unc = 6

z_names = ['name', 'D13_N2S2_O3S2', 'ne_D13_N2S2_O3S2', 'pe_D13_N2S2_O3S2', 'KD02comb', 'ne_KD02comb', 'pe_KD02comb','PP04_O3N2', 'ne_PP04_O3N2','pe_PP04_O3N2', 'M08_N2Ha', 'ne_M08_N2Ha', 'pe_M08_N2Ha','E(B-V)', 'ne_E(B-V)', 'pe_E(B-V)','M13_O3N2', 'ne_M13_O3N2', 'pe_M13_O3N2']

data = ascii.read(fp + filename, names=z_names)

ptf_unc = []
ptf_icbl = []
ptf_ic = []

for i in range(0,len(data['name'])):
    if i < no_unc:
        ascii.write(data[z_names][0:no_unc], 'ptf_unc_z.txt', names = z_names)
    elif i < no_unc+no_icbl:
        ascii.write(data[z_names][no_unc+1:no_unc+no_icbl], 'ptf_icbl_z.txt', names = z_names)
    else:
        ascii.write(data[z_names][no_unc+no_icbl+1:len(data)], 'ptf_ic_z.txt', names = z_names)
        



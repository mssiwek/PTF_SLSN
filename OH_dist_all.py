#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:51:20 2018

@author: magda
"""

#metallicity distribution of SLSNe, SN-GRBs, SNe Ic-BL and Ic

import numpy as np
from astropy.io import ascii
import os
import pathlib
import matplotlib.pyplot as plt
import scipy.stats as ss
import matplotlib.gridspec as gridspec
from scipy import stats

#below are the keys for the columns in the metallicity tables
z_names = ['name', 'D13_N2S2_O3S2', 'ne_D13_N2S2_O3S2', 'pe_D13_N2S2_O3S2', 'KD02comb', 'ne_KD02comb', 'pe_KD02comb','PP04_O3N2', 'ne_PP04_O3N2','pe_PP04_O3N2', 'M08_N2Ha', 'ne_M08_N2Ha', 'pe_M08_N2Ha','E(B-V)', 'ne_E(B-V)', 'pe_E(B-V)','M13_O3N2', 'ne_M13_O3N2', 'pe_M13_O3N2']

#file path extensions
slsn_z = '/slsn_i_z.txt'
grb_z ='/grboh_083017.txt'
ic_z = '/ptf_ic_z.txt'
icbl_z ='/ptf_icbl_z.txt'

#define variable for fontsize
fs = 10

#current directory
fp = os.getcwd()
#go two paths up, then into the folder 'data' where the above files are stored
fp2 = str(pathlib.Path(__file__).parents[2]) + '/data/'

inslsn1 = 'apjaa3522t1_mrt.txt'
slsn_t1 = ascii.read(inslsn1)

#note: slsn does not use names=z_names because not all metallicity calibrations are available in this table
#slsn only has: KD02comb, PP04_O3N2, M08_N2Ha, M13_O3N2

slsn = ascii.read(fp+slsn_z)
grb = ascii.read(fp+grb_z, names=z_names)
ic = ascii.read(fp+ic_z, names=z_names)
icbl = ascii.read(fp+icbl_z, names=z_names)

colors = ['black','#9400D3','#386890','IndianRed','#FFD700']
linestyles = ['dotted', 'dashed', 'solid', 'dotted']
          
m = 0
left  = 0.2  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.2   # the bottom of the subplots of the figure
top = 0.7    # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.2   # the amount of height reserved for space between subplots,
               # expressed as a fraction of the average axis height
fig = plt.figure(figsize=(20, 15))
outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
alpha = 0.2
#make the histograms for all scales
for z_scale in ['KD02comb', 'PP04_O3N2', 'M08_N2Ha', 'M13_O3N2']:
    m+=1
    
    #find where values are missing and delete them from the array since otherwise 
    #there will be errors when converting to float
    ne_index_slsn = np.where(slsn['ne_'+z_scale] == '...')
    pe_index_slsn = np.where(slsn['pe_'+z_scale] == '...')
    index_slsn = np.where(slsn[z_scale] == 0.0)
    slsn2 = np.delete(slsn[z_scale], index_slsn)
    slsn['ne_'+z_scale][ne_index_slsn] = 0 
    slsn['pe_'+z_scale][pe_index_slsn] = 0
    ne_slsn = np.delete(slsn['ne_'+z_scale],index_slsn).astype(np.float)
    pe_slsn = np.delete(slsn['pe_'+z_scale], index_slsn).astype(np.float)
    
    #find where values are missing and delete them from the array since otherwise 
    #there will be errors when converting to float
    ne_index_grb = np.where(grb['ne_'+z_scale] == '...')
    pe_index_grb = np.where(grb['pe_'+z_scale] == '...')
    index_grb = np.where(grb[z_scale] == '...')
    grb2 = np.delete(grb[z_scale], index_grb)
    grb['ne_'+z_scale][ne_index_grb] = 0
    grb['pe_'+z_scale][pe_index_grb] = 0
    ne_grb = np.delete(grb['ne_'+z_scale],index_grb).astype(np.float)
    pe_grb = np.delete(grb['pe_'+z_scale], index_grb).astype(np.float)
    
    #find where values are missing and delete them from the array since otherwise 
    #there will be errors when converting to float
    ne_index_ic = np.where(ic['ne_'+z_scale] == '...')
    pe_index_ic = np.where(ic['pe_'+z_scale] == '...')
    index_ic = np.where(ic[z_scale] == '...')
    ic2 = np.delete(ic[z_scale], index_ic)
    ic['ne_'+z_scale][ne_index_ic] = 0
    ic['pe_'+z_scale][pe_index_ic] = 0
    ne_ic = np.delete(ic['ne_'+z_scale],index_ic).astype(np.float)
    pe_ic = np.delete(ic['pe_'+z_scale],index_ic).astype(np.float)
    
    #find where values are missing and delete them from the array since otherwise 
    #there will be errors when converting to float
    index_icbl = np.where(icbl[z_scale] == '...')
    ne_index_icbl = np.where(icbl['ne_'+z_scale] == '...')
    pe_index_icbl = np.where(icbl['pe_'+z_scale] == '...')
    icbl2 = np.delete(icbl[z_scale], index_icbl)
    icbl['ne_'+z_scale][ne_index_icbl] = 0
    icbl['pe_'+z_scale][pe_index_icbl] = 0
    ne_icbl = np.delete(icbl['ne_'+z_scale],index_icbl).astype(np.float)
    pe_icbl = np.delete(icbl['pe_'+z_scale], index_icbl).astype(np.float)
    
    #put all the data in one array
    all_data = [ic2.astype(np.float), slsn2.astype(np.float), icbl2.astype(np.float), grb2.astype(np.float)]
    all_pe_errors = [pe_ic.astype(np.float), pe_slsn.astype(np.float), pe_icbl.astype(np.float), pe_grb.astype(np.float)]
    all_ne_errors = [ne_ic.astype(np.float), ne_slsn.astype(np.float), ne_icbl.astype(np.float), ne_grb.astype(np.float)]
    all_labels = ['Ic: %d' %len(all_data[0]),  'SLSN Ic: %d' %len(all_data[1]), 'Ic-BL: %d' %len(all_data[2]), 'SN-GRB: %d' %len(all_data[3])]
    
    #get the bins, first for the data
    '''
    binwidth = 0.08
    slsn_bins=np.arange(float(min(slsn2)), float(max(slsn2)) + binwidth, binwidth)
    grb_bins = np.arange(float(min(grb2)), float(max(grb2)) + binwidth, binwidth)
    ic_bins = np.arange(float(min(ic2)), float(max(ic2)) + binwidth, binwidth)
    icbl_bins = np.arange(float(min(icbl2)), float(max(icbl2)) + binwidth, binwidth)
    bins = [ic_bins, slsn_bins, icbl_bins, grb_bins]
    '''
    x_range = [7.25, 9.5]
    num_bins = len(np.arange(x_range[0], x_range[1], 0.1))
    bins = [num_bins,num_bins,num_bins,num_bins]
    '''
    #now for the errors
    ne_slsn_bins=np.arange(float(min(slsn2-ne_slsn)), float(max(slsn2-ne_slsn)) + binwidth, binwidth)
    ne_grb_bins = np.arange(float(min(grb2-ne_grb)), float(max(grb2-ne_grb)) + binwidth, binwidth)
    ne_ic_bins = np.arange(float(min(ic2-ne_ic)), float(max(ic2-ne_ic)) + binwidth, binwidth)
    ne_icbl_bins = np.arange(float(min(icbl2-ne_icbl)), float(max(icbl2-ne_icbl)) + binwidth, binwidth)
    ne_bins = [ne_ic_bins, ne_slsn_bins, ne_icbl_bins, ne_grb_bins]
    
    pe_slsn_bins=np.arange(float(min(slsn2+pe_slsn)), float(max(slsn2+pe_slsn)) + binwidth, binwidth)
    pe_grb_bins = np.arange(float(min(grb2+pe_grb)), float(max(grb2+pe_grb)) + binwidth, binwidth)
    pe_ic_bins = np.arange(float(min(ic2+pe_ic)), float(max(ic2+pe_ic)) + binwidth, binwidth)
    pe_icbl_bins = np.arange(float(min(icbl2+pe_icbl)), float(max(icbl2+pe_icbl)) + binwidth, binwidth)
    pe_bins = [pe_ic_bins, pe_slsn_bins, pe_icbl_bins, pe_grb_bins]
    '''
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[m-1], wspace=0.1, hspace=0.)

    ax1 = plt.Subplot(fig, inner[0])
    ax2 = plt.Subplot(fig, inner[1], sharex=ax1)
    
    for i in range(0, len(all_data)):
        
        x_range = [7.25, 9.5]
        num_bins = 8#len(np.arange(x_range[0], x_range[1], 0.2))
        
        binwidth = 0.1
        slsn_bins=np.arange(float(min(slsn2)), float(max(slsn2)) + binwidth, binwidth)
        grb_bins = np.arange(float(min(grb2)), float(max(grb2)) + binwidth, binwidth)
        ic_bins = np.arange(float(min(ic2)), float(max(ic2)) + binwidth, binwidth)
        icbl_bins = np.arange(float(min(icbl2)), float(max(icbl2)) + binwidth, binwidth)
        bins = [ic_bins, slsn_bins, icbl_bins, grb_bins]
        
        ax1.hist(all_data[i], bins =bins[i], color = colors[i], label=all_labels[i])
        ax1.legend(fontsize = fs-1)
        #ax1.set_xticklabels(fontsize=fs)
        #ax1.set_xticklabels([])
        ax1.set_ylabel('Number of hosts', fontsize=fs)
        x_range = [7.25, 9.5]
        ax1.tick_params(axis='x',which='major',direction='in',length=10,width=1,pad=10,labelcolor='w')
        ax1.tick_params(axis='x',which='minor',direction='in',length=5,width=1,pad=10, labelcolor='w')
        ax1.set_xlim(x_range[0], x_range[1])
        ax1.tick_params(axis='y',which='major',direction='in',length=10,width=1,pad=10, labelsize=fs)
        ax1.tick_params(axis='y',which='minor',direction='in',length=5,width=1,pad=10, labelsize=fs)
        ax1.minorticks_on()
        #ax1.set_ylim([0,np.ceil(np.max(all_data[i]))])
        #ax2.set_xticklabels([7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25])
        #n, binns, patches = plt.hist(all_data[i], color = colors[i], label=all_labels[i], histtype='step', linestyle='dashed', cumulative=True, normed=True, linewidth=2)

        '''
        nx, xbins = np.histogram(all_data[i], bins=bins[i], normed=1, density=True)#, cumulative=True)
        xtemp4 = [xbins[0]]
        xtemp4.extend(xbins[:-1])
        ytemp4 = [0]
        ytemp4.extend(nx.cumsum().astype(float)/nx.sum())
        
        #now the errors
        #lower end errors
        nx, xbins= np.histogram((all_data[i] - all_ne_errors[i]), bins=bins[i], normed=1, density=True)#, cumulative=True)
        xtemp4a = [xbins[0]]
        xtemp4a.extend(xbins[:-1])
        ytemp4a = [0]
        ytemp4a.extend(nx.cumsum().astype(float)/nx.sum())
        
        #upper end errors
        nx, xbins= np.histogram((all_data[i]+ all_pe_errors[i]), bins=bins[i], normed=1, density=True)#, cumulative=True)
        xtemp4b = [xbins[0]]
        xtemp4b.extend(xbins[:-1])
        ytemp4b = [0]
        ytemp4b.extend(nx.cumsum().astype(float)/nx.sum())
        
        ax2.plot(xtemp4, ytemp4, linestyle='dashed',color=colors[i])
        #ax2.plot(xtemp4a, ytemp4a, color=colors[i])
        #ax2.plot(xtemp4b, ytemp4b, color=colors[i])
        ax2.fill_between(xtemp4, ytemp4a, ytemp4b, color=colors[i], alpha=0.4)
        #ax2.fill_betweenx(ytemp4a, xtemp4a,xtemp4b, color=colors[i], alpha=0.4)
        #ax2.fill(xtemp4a, ytemp4a, colors[i],alpha=0.4)
        '''
        
        binsize = 0.0021
        num_bins = int((x_range[1]-x_range[0])/binsize)
        
        n, bins, patches = ax2.hist(all_data[i], bins=num_bins,
                                                    range=x_range,
                                                    histtype='step',
                                                    color=colors[i],
                                                    linestyle=linestyles[i],
                                                    cumulative=True,
                                                    normed=True, linewidth=2)
        

        lohs = all_data[i] - all_ne_errors[i]
        temp, bins = np.histogram(lohs, bins=num_bins,range=x_range, density=True)

        temp = np.cumsum(temp)
        bins = bins[0:-1] # drop the upper edge

        # normalize??
        lower = temp / float(max(temp))

        uohs = all_data[i] + all_pe_errors[i]
        temp, bins = np.histogram(uohs, bins=num_bins,range=x_range, density=True)
        bins = bins[0:-1] # drop the upper edge
        temp = np.cumsum(temp)
        upper = temp/float(max(temp))

        # ok, fill in the low-up limit region
        ax2.fill_between(bins, lower, upper,color=colors[i], alpha=alpha)


        # lighter shades NOW DARKER
        # loop over SN type

        vals = all_data[i]
        #sortint it 
        vals = sorted(vals)
        medians = []


        #looping over N measurements = N SNe
        # here is the drop 1
        for k, val in enumerate(vals):
            # ok, removed one
            tempvals = vals[0:k]+vals[(k+1):]
            # FBB: here it is: you are taking the means along the rows! you want to get the cumulative hist, then tke the means of those

            n, bins = np.histogram(tempvals,bins=num_bins,range=x_range)
            n = n.cumsum().astype(float) / n.sum()
            #print(n)

            medians.append(n)

            #pdb.set_trace()

    
                                    
        #print (len(medians), len(n))
        meanhere = np.array(medians).mean(axis=0)
        stdhere = np.array(medians).std(axis=0)
        semhere = stats.sem(np.array(medians), axis=0)



        lower = (meanhere - stdhere).tolist()
        upper = (meanhere + stdhere).tolist()                        
        print(lohs)

        bins = bins[0:-1] # drop the upper edge

        ax2.fill_between(bins, lower, upper, color=colors[i], alpha=alpha * 2.)



        
        
        ax2.set_ylabel('Fraction of SNe', fontsize=fs)
        ax2.set_ylim([0,1])
        ax2.set_xlabel('log(O/H)+12 (%s)' %z_scale, fontsize=fs)
        ax2.set_xticks(np.linspace(x_range[0], x_range[1], 10))
        ax2.minorticks_on()
        ax2.tick_params(axis='x',which='major',direction='in',length=10,width=1,pad=10, labelsize=fs)
        ax2.tick_params(axis='x',which='minor',direction='in',length=5,width=1,pad=10, labelsize=fs)
        ax2.set_xlim(x_range[0], x_range[1])
        ax2.tick_params(axis='y',which='major',direction='in',length=10,width=1,pad=10, labelsize=fs)
        ax2.tick_params(axis='y',which='minor',direction='in',length=5,width=1,pad=10, labelsize=fs)
        
        
        
        
        
        
        
        
        
        
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    
plt.savefig(fp + '/' + 'OH_dist_062918_only_I.jpg')
plt.close()





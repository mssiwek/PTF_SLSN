"""
Generate various scatter point plots, e.g., Z vs. M, from SED fitting and pyMCZ results
"""
import math
import pickle as pkl
import re
# from astropy import cosmology
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io.idl import readsav
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from astropy.io import ascii
from matplotlib.ticker import FormatStrFormatter

sncolors = {"Ic": 'black',
            "Ic-BL":'#386890',
            "GRB":'IndianRed',
            "sdss":'#FFD700', "slsn":'#9400D3'}

# #9400D3 is a dark violet. check: https://www.webucator.com/blog/2015/03/python-color-constants-module/


def scatterfit(x, y, a=None, b=None):

    if a == None:
        a, b, r, p, err = stats.linregress(x, y)
    N = np.size(x)
    sd = 1./(N-2.)*np.sum((y-a*x-b)**2); sd = np.sqrt(sd)
    return sd

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def squared(x, mu, sig):
    ys = np.zeros(x.shape[0])
    for i, y in enumerate(ys):
        if abs(x[i]-mu) < sig: ys[i] = 1
    return ys

def avgtrend(xs, ys, smooth):
    # trim the nan cases:
    xdata = []
    ydata = []
    for i, y in enumerate(ys):
        if not np.isnan(y):
            xdata.append(xs[i])
            ydata.append(y)
    num = 20 # avg trend got calculated at num data points
    sig = (max(xdata)-min(xdata))/num*smooth
    print('smoothing scale for average trend', sig)
    xavgs = np.linspace(min(xdata), max(xdata), num=num)
    yavgs = np.zeros(xavgs.shape[0])
    for i, xavg in enumerate(xavgs):
        xweights = np.array(xdata)
        yweights = gaussian(xweights, xavg, sig)
        #yweights = squared(xweights, xavg, sig)
        yavgs[i] = np.average(ydata, weights=yweights)

    return xavgs, yavgs

def avgtrend2(xs, ys):
    # trim the nan cases:
    xydata = []
    xdata = []
    for i, y in enumerate(ys):
        if not np.isnan(y):
            xdata.append(xs[i])
            xydata.append((xs[i], y))
    num = 10 # avg trend got calculated at num data points
    xavgs = np.linspace(min(xdata), max(xdata), num=num)
    yavgs = np.zeros(xavgs.shape[0])
    binsize = (max(xdata)-min(xdata))/(num-1)
    # decide in which bin each data sit
    for i, xavg in enumerate(xavgs):
        yavgs[i] = np.median([x[1] for x in xydata if x[0]>xavg-binsize/2 and
                              x[0]<=xavg+binsize/2])

    return xavgs, yavgs

def avgtrend3(xs, ys):
    # trim the nan cases:
    xdata = []
    ydata = []
    for i, y in enumerate(ys):
        if not np.isnan(y):
            xdata.append(xs[i])
            ydata.append(y)

    num = 2 # avg trend got calculated at num data points
    xdata = np.array(xdata).reshape(len(xdata), -1)
    ydata = np.array(ydata).reshape(len(ydata), -1)
    linreg = LinearRegression().fit(xdata, ydata)
    #print 'intercept:', linreg.intercept_
    #print 'coeff:', linreg.coef_
    #print 'sample size', len(xdata)

    xavgs = np.linspace(min(xdata), max(xdata), num=num).reshape(num, -1)
    return xavgs, linreg.predict(xavgs)

def avgtrend4(xs, ys):
    # trim the nan cases:
    xdata = []
    ydata = []
    for i, y in enumerate(ys):
        if not np.isnan(y):
            xdata.append(xs[i])
            ydata.append(y)

    num = 2 # avg trend got calculated at num data points
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    a, b, r, p, err = stats.linregress(xdata, ydata)
    #print('intercept:', b)
    #print('coeff:', a)
    #print 'sample size', len(xdata)

    # confidence interval
    alpha = 0.05
    n = len(xdata)
    x = np.linspace(min(xdata), max(xdata), num=100)
    y = b + a * x
    sd = scatterfit(xdata, ydata, a, b)
    sxd = np.sum((xdata - xdata.mean())**2.)
    sx = (x-xdata.mean())**2.

    q = stats.t.ppf(1. - alpha/2., n-2)

    dy = q*sd*np.sqrt(1./n + sx/sxd)
    ucb = y+dy
    lcb = y-dy
    return lcb, ucb, x

#don't want to use this as a function, i want to be able to check the variables
#def OH_MB(inpymcz, infit, idfile, inlvl, ingrb1, ingrb2, insdss, insfsq, inohsq, insSFRsq, outpdf, outpkl, inslsn1, inslsn2, inslsn3, inslsn4, slsn_z):

inpymcz = 'platefitflux_072617.pkl'
infit = 'SED_all_072117.pkl'
idfile = 'PTF_SDSS_072117.txt'
inlvl = 'sdss/lvlsz.sav'
ingrb1 = 'GRB_083017.pkl'
ingrb2 = 'GRBMsSFR_083017.txt'
insdss = 'sdss/intervals.txt'
insfsq = 'sdss/sfsq.txt'
inohsq = 'sdss/ohsq.txt'
insSFRsq = 'sdss/sSFRsq.txt'
outpdf = 'OH_MB_061118_all_'
outpkl = 'OH_MB_121217_all'
inslsn1 = 'apjaa3522t1_mrt.txt' #contains coordinates, redshift, magnitude, time of SN
inslsn2 = 'apjaa3522t2_mrt.txt' #contains magnitude, flux density, observation date, instrument
inslsn3 = 'apjaa3522t3_mrt.txt' #contains observation date, flag, setup, exposure sequence, sky position angle
inslsn4 = 'apjaa3522t4_mrt.txt' #contains integrated galaxy magnitude, SFR, galaxy mass, extinction
inslsn5_orig = 'apjaa3522t5_mrt.txt' #contains integrated galaxy magnitude, SFR, galaxy mass, extinction
slsn_i_z = 'slsn_i_z.txt'

mczs = pkl.load(open(inpymcz, 'rb'))
PTFnames = [x["PTFname"] for x in mczs]
fits = pkl.load(open(infit, 'rb'))
f = open(idfile, 'r')
text = f.read()
f.close()
match_all = re.findall(r'(\S+)\s\s(\S+)', text)
PTFs = [x[0] for x in match_all]
objids = [x[1] for x in match_all]

temp = readsav(inlvl)
lvls = temp['lvls']

grbs1 = pd.read_pickle(ingrb1)
# the new SN-GRB 2016jca is in ingrb1 but not ingrb2 (stellar mass and SFR not in GHost)
#grbs1 = grbs1[:-1]
GRBnames = [x['PTFname'] for x in grbs1]

grbs2 = pd.read_csv(ingrb2, skiprows=2, delim_whitespace=True, header=None,
                    index_col='galnum',
                    names=['galnum', 'reference', 'hostname_and_region',
                           'logMs', 'elogMs', 'SFR',
                           'eSFR', 'logMs_L', 'elogMs_L'], na_values='-')
#grbs2 = grbs2[0:9]
#grbs2 = grbs2[0:10]
grbs2 = grbs2[0:11]
grbs2['hostname'] = grbs2['hostname_and_region'].str.split('-').str[0].str[-8:]
# fix the name for the one with different naming convension
grbs2.loc['3']['hostname'] = 'XRF020903'
grbs2.loc['10']['hostname'] = 'SN2016jca'
grbs2['region'] = 'SNsite'
# change region of SN2003lw from SNsite to other so that got plotted with open symbol
grbs2.loc[grbs2['hostname'] == 'SN2003lw', 'region'] = 'other'
grbs2.loc[grbs2['hostname'] == 'SN2009nz', 'region'] = 'other'
grbs2['elogSFR'] = np.array(grbs2['eSFR']).astype(np.float) /\
                   (np.array(grbs2['SFR']).astype(np.float)*np.log(10.))
#grbs1= grbs1[~(grbs1['hostname'] == 'SN2009nz')]

#print(grbs1)

#temp = pd.read_pickle(in12gzk1)
#gzk1 = [x for x in temp if x['PTFname'] == '12gzk']

#gzk2 = pd.read_pickle(in12gzk2)

for grb1 in grbs1:
    temp = grbs2[grbs2['hostname'] == grb1['PTFname']]['logMs']
    #print temp
    grb1['logMs'] = temp.iloc[0]
    temp = grbs2[grbs2['hostname'] == grb1['PTFname']]['elogMs']
    grb1['elogMs'] = temp.iloc[0]
    temp = grbs2[grbs2['hostname'] == grb1['PTFname']]['SFR']
    grb1['SFR'] = temp.iloc[0]
    temp = grbs2[grbs2['hostname'] == grb1['PTFname']]['elogSFR']
    grb1['elogSFR'] = temp.iloc[0]
    temp = grbs2[grbs2['hostname'] == grb1['PTFname']]['region']
    grb1['region'] = temp.iloc[0]
# print [x["region"] for x in grbs1]

# make overlapped sample
clean = []
for i, fit in enumerate(fits):
    objid = str(fit["sloan_objid"])
    objid = objid.replace(" ", "")
    PTF = PTFs[objids.index(objid)]
    mczindex = PTFnames.index(PTF)
    clean.append(mczs[mczindex])

# calculate B-band absolute magnitudes
zs = [x["redshift"] for x in fits]
z = np.array(zs)
# dms = [cosmology.distmod(z, H0=70, Om=0.3, Ol=0.7, Ok=0) for z in zs]
c = 299792.45
H=70.
dm = 25.+5.*np.log10(c*z/H)
dist = c*z/H
gmag = np.array([x["modelmag"][1] for x in fits])
rmag = np.array([x["modelmag"][2] for x in fits])
M_B = gmag + 0.3130*(gmag - rmag) + 0.2271 - dm
M_g = gmag - dm
M_r = rmag - dm

for i, gal in enumerate(clean):
    gal["M_B"] = M_B[i]
    gal["M_g"] = M_g[i]
    gal["M_r"] = M_g[i]
    gal["dist"] = dist[i]
    gal["redshift"] = z[i]
    gal["logMs"] = fits[i]["LgMs"]
    gal["logSFR"] = fits[i]["LgSFR"]
    gal["logsSFR"] = fits[i]["LgSFR_spec"]
    gal["modelmag"] = fits[i]["modelmag"]
    gal["modelerr"] = fits[i]["modelerr"]

fig = plt.figure(figsize=(15, 12))
fs = 20
ls = 20
#plt.xlabel('Distance [Mpc]', fontsize=fs)
plt.xlabel(r'${\rm redshift}$', fontsize=fs)
plt.ylabel(r'${\rm Absolute\ {\it r}-band\ magnitude}$', fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=ls)
#x_range = (0, 700)
x_range = (0., 0.51)
y_range = (-14.5, -24.5)
ax = fig.add_subplot(111)
plt.xlim(x_range[0], x_range[1])
plt.ylim(y_range[0], y_range[1])
plt.minorticks_on()
plt.setp(ax.yaxis.get_ticklines(), 'markersize', 7)
plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 1)
plt.setp(ax.xaxis.get_ticklines(), 'markersize', 7)
plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 1)
plt.setp(ax.yaxis.get_ticklines(minor=True), 'markersize', 4)
plt.setp(ax.yaxis.get_ticklines(minor=True), 'markeredgewidth', 1)
plt.setp(ax.xaxis.get_ticklines(minor=True), 'markersize', 4)
plt.setp(ax.xaxis.get_ticklines(minor=True), 'markeredgewidth', 1)
#temp7 = [x[5] for x in lvls]
#temp8 = [x[6] for x in lvls]
#temp9 = [x[9]*50 for x in lvls]
#plt.scatter(temp7, temp8, s=temp9, color='grey')

temp1 = [x["redshift"] for x in clean if x["SNtype"] == "Ic" and x["region"] != "SNsite"]
temp2 = [x["M_r"] for x in clean if x["SNtype"] == "Ic" and x["region"] != "SNsite"]
plt.plot(temp1, temp2, color = sncolors['Ic'], marker = 's', linestyle='None', fillstyle='none', label='SN Ic (nuc or nearby HII)',
         mew=1.5, ms=10)

temp1 = [x["redshift"] for x in clean if x["SNtype"] == "Ic" and x["region"] == "SNsite"]
temp2 = [x["M_r"] for x in clean if x["SNtype"] == "Ic" and x["region"] == "SNsite"]
plt.plot(temp1, temp2, color = sncolors['Ic'], marker = 's',linestyle='None', label='SN Ic (SN site)', mew=1.5, ms=10,
         markeredgecolor='none')

temp1 = [x["redshift"] for x in clean if x["SNtype"] == "Ic-BL" and x["region"] != "SNsite"]
temp2 = [x["M_r"] for x in clean if x["SNtype"] == "Ic-BL" and x["region"] != "SNsite"]
plt.plot(temp1, temp2, color = sncolors['Ic-BL'], marker = 'o', linestyle='None',fillstyle='none', label='SN Ic-BL (nuc or nearby HII)',
         mew=1.5, ms=10)

temp1 = [x["redshift"] for x in clean if x["SNtype"] == "Ic-BL" and
         x["region"] == "SNsite"]
temp2 = [x["M_r"] for x in clean if x["SNtype"] == "Ic-BL" and x["region"] == "SNsite"]
plt.plot(temp1, temp2, color = sncolors['Ic-BL'], marker = 'o', linestyle='None',label='SN Ic-BL (SN site)', mew=1.5, ms=10,
         markeredgecolor='none')

temp1 = [x["redshift"] for x in clean if x["SNtype"] == "uncertain"]
temp2 = [x["M_r"] for x in clean if x["SNtype"] == "uncertain"]
plt.plot(temp1, temp2, 'g^', label='uncertain/weird SN subtype', mew=1.5, ms=10, fillstyle='none')

#now the SLSN:
#this is table 4 from perley et al, which contains (amongst other things)
#the redshift and visual magnitude (which I think is in g-band...)

slsn_t1 = ascii.read(inslsn1)

#find the indeces at which the slsn hosts are type I or I-R, but NOT II:
type_i_slsn = np.where(slsn_t1['Class'] != 'II')

#now only extract the type I from each table we are using
slsn_t1 = slsn_t1[type_i_slsn]

slsn_t4 = ascii.read(inslsn4)
slsn_t4 = slsn_t4[type_i_slsn]

#but want to use the separate file to get only type I in table 6 because it contains
#duplicates and indexing doesn't work
slsn_z_dat = ascii.read(slsn_i_z)

temp1 = slsn_t1['z']
temp2 = slsn_t4['Mg']
plt.plot(temp1, temp2, color = sncolors['slsn'], marker = '*', linestyle='None',label='SLSN Ic host magnitude (g-band?)', mew=1.5, ms=10, markeredgecolor='none')

print("Median redshift of Ic hosts: ", np.median([x["redshift"] for x in clean
                                                  if x["SNtype"] == "Ic"]))
print("Median redshift of Ic-BL hosts: ", np.median([x["redshift"] for x in clean
                                                     if x["SNtype"] == "Ic-BL"]))
print("Average redshift of Ic hosts: ", np.average([x["redshift"] for x in clean
                                                    if x["SNtype"] == "Ic"]))
print("Average redshift of Ic-BL hosts: ", np.average([x["redshift"] for x in clean
                                                       if x["SNtype"] == "Ic-BL"]))
print("Standard deviation of redshift (Ic): ", np.std([x["redshift"] for x in clean
                                                       if x["SNtype"] == "Ic"]))
print("Standard deviation of redshift (Ic-BL): ", np.std([x["redshift"] for x in clean
                                                          if x["SNtype"] == "Ic-BL"]))

temp3 = np.linspace(x_range[0]+0.01, x_range[1], 2000)
temp4 = 17.77 - (25.+5.*np.log10(c*temp3/H))
plt.plot(temp3, temp4, '-.', color=sncolors['sdss'], label='SDSS magnitude limit',
         linewidth=2.5)

legend = plt.legend(loc='lower right', fontsize=fs-5, frameon=False,
                    numpoints=1)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#ax.set_xticklabels(np.arange(0.,0.51,0.05))
#ax.set_xticks(np.arange(0.,0.51,0.05))

outfile = outpdf+'0.pdf'
plt.savefig(outfile, format='pdf')
plt.close(fig)

# x_range = (-15., -23.5) for B band absolute magnitude
x_range = (6.5, 12.2)
y_range = (7.3, 9.6)

fig = plt.figure(figsize=(10, 10))
fs = 20
ls = 15
fig.subplots_adjust(left=0.1, bottom=0.18, hspace=0, wspace=0)

#OHscales = ['D13_N2S2_O3S2', 'KD02comb', 'M13_O3N2', 'M08_N2Ha']
OHscales = ['KD02comb', 'PP04_O3N2', 'M08_N2Ha', 'M13_O3N2']

# read in SDSS trends
sdss = {}
file4 = open(insdss, 'r')
lines = file4.readlines()
flag = 0
for line in lines:
    if re.search(r'[a-zA-Z]+\d+_[a-zA-Z]+', line):
        key = line[:-1]
        flag = 0
    if line.startswith('['):
        line = line[1:-2]
        vals = line.split(r', ')
        vals = [float(val) for val in vals]
        flag = 1
    if flag:
        sdss[key] = vals
file4.close()

for i, OHscale in enumerate(OHscales):
    ax = fig.add_subplot(221+i)
    # plt.title('Host Galaxy', fontsize=fs+2)
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.minorticks_on()
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 7)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 1)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 7)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 1)
    plt.setp(ax.yaxis.get_ticklines(minor=True), 'markersize', 4)
    plt.setp(ax.yaxis.get_ticklines(minor=True), 'markeredgewidth', 1)
    plt.setp(ax.xaxis.get_ticklines(minor=True), 'markersize', 4)
    plt.setp(ax.xaxis.get_ticklines(minor=True), 'markeredgewidth', 1)
    plt.tick_params(axis='both', which='major', labelsize=ls)
    if i < 2:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        #plt.xlabel('Host Galaxy M_B (mag)', fontsize=fs)
        plt.xlabel(r'$\log\ M_* {\rm\ [ M_{\odot}]}$', fontsize=fs)
    if i % 2 == 1:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
    else:
        plt.ylabel(r'$\log\ {\rm(O/H)+12}$', fontsize=fs)
    plt.annotate(OHscale, xy=(0.6, 0.15), xycoords='axes fraction', fontsize=fs-7)


    # drop the ones OHscale na
    clean1 = []
    for cleanone in clean:
        if OHscale in cleanone.keys():
            clean1.append(cleanone)

    temp7_lvl = [np.log10(x[11]) for x in lvls]
    temp8_lvl = [x[12] for x in lvls]
    temp9_lvl = [x[9]*50 for x in lvls]
    point0 = plt.scatter(temp7_lvl, temp8_lvl, s=temp9_lvl, color='lightgrey', label='LVL')
    #print OHscale
    #print temp8_lvl
    
    #xavgs, yavgs = avgtrend(temp7, temp8, smooth=1)

    temp1 = [x["logMs"][0] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] != "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] != "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] != "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] != "SNsite"]
    temp5 = [x["logMs"][0]-x["logMs"][3] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] != "SNsite"]
    temp6 = [x["logMs"][4]-x["logMs"][0] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] != "SNsite"]
    point1 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                          fmt='ks', fillstyle='none', label='SN Ic (nuc or nearby HII)')

    temp1 = [x["logMs"][0] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] == "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] == "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] == "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] == "SNsite"]
    temp5 = [x["logMs"][0]-x["logMs"][3] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] == "SNsite"]
    temp6 = [x["logMs"][4]-x["logMs"][0] for x in clean1 if x["SNtype"] == "Ic"
             and x["region"] == "SNsite"]
    point2 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                          fmt='ks', markeredgecolor='none', label='SN Ic (SN site)')

    temp1 = [x["logMs"][0] for x in clean1 if x["SNtype"] == "Ic"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"]
    xavgs, yavgs = avgtrend3(temp1, temp2)
    plt.plot(xavgs, yavgs, color=sncolors['Ic'], linestyle='dotted', linewidth=2)
    #if OHscale == 'PP04_O3N2':
    #       lcb, ucb, xavgs = avgtrend4(temp1, temp2)
    #       plt.fill_between(xavgs, lcb, ucb, color='black', alpha=0.2)

    temp1 = [x["logMs"][0] for x in clean1 if x["SNtype"] == "Ic-BL" and x["region"] != "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL" and x["region"] != "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic-BL"
             and x["region"] != "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and x["region"] != "SNsite"]
    temp5 = [x["logMs"][0]-x["logMs"][3] for x in clean1 if x["SNtype"] == "Ic-BL"
             and x["region"] != "SNsite"]
    temp6 = [x["logMs"][4]-x["logMs"][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and x["region"] != "SNsite"]
    point3 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                          fmt = 'o', color=sncolors['Ic-BL'], fillstyle='none',
                          label='SN Ic-BL (nuc or nearby HII)')

    temp1 = [x["logMs"][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and x["region"] == "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and x["region"] == "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic-BL"
             and x["region"] == "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and x["region"] == "SNsite"]
    temp5 = [x["logMs"][0]-x["logMs"][3] for x in clean1 if x["SNtype"] == "Ic-BL"
             and x["region"] == "SNsite"]
    temp6 = [x["logMs"][4]-x["logMs"][0] for x in clean1
             if x["SNtype"] == "Ic-BL" and x["region"] == "SNsite"]
    point4 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                          fmt = 'o', color=sncolors['Ic-BL'], markeredgecolor='none',
                          label='SN Ic-BL (SN site)')

    temp1 = [x["logMs"][0] for x in clean1 if x["SNtype"] == "Ic-BL"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"]
    xavgs, yavgs = avgtrend3(temp1, temp2)
    plt.plot(xavgs, yavgs, color=sncolors['Ic-BL'], linestyle='dashed', linewidth=2)
    #if OHscale == 'PP04_O3N2':
    #       lcb, ucb, xavgs = avgtrend4(temp1, temp2)
    #       plt.fill_between(xavgs, lcb, ucb, color='blue', alpha=0.2)

    grbs_clean = []


    grbs1 = np.array(grbs1)[~np.array([grb["PTFname"] == "SN2009nz" for grb in grbs1])]
    for grb1 in grbs1:
        if OHscale in grb1.keys():
            if grb1["PTFname"] == "SN2013dx":
                grb1[OHscale][1] = grb1[OHscale][0]
                grb1[OHscale][2] = grb1[OHscale][0]
                grb1["upperlimit"] = 1
            elif grb1["PTFname"] == "SN2009nz":
                continue
                '''
                if OHscale in ['PP04_O3N2', 'M08_N2Ha']:
                        grb1[OHscale][1] = grb1[OHscale][0]
                        grb1[OHscale][2] = grb1[OHscale][0]
                        grb1["upperlimit"] = 1
                else:
                        grb1["upperlimit"] = 0
                '''
            else:
                grb1["upperlimit"] = 0
            grbs_clean.append(grb1)

    temp1 = [float(x['logMs']) for x in grbs_clean if x['region'] != 'SNsite']
    temp2 = [x[OHscale][0] for x in grbs_clean if x['region'] != 'SNsite']
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in grbs_clean
             if x['region'] != 'SNsite']
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in grbs_clean
             if x['region'] != 'SNsite']
    temp5 = [float(x['elogMs']) for x in grbs_clean if x['region'] != 'SNsite']
    temp6 = [float(x['elogMs']) for x in grbs_clean if x['region'] != 'SNsite']
    temp10 = np.array([x['upperlimit'] for x in grbs_clean
                       if x['region'] != 'SNsite'], dtype=bool)
    if len(temp1) > 0:
        point5 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6],
                              yerr=[temp3, temp4], fmt='D', color=sncolors['GRB'],
                              fillstyle='none', label='SN-GRB (wAGN)')
    if True in temp10:
        x1 = [float(x['logMs']) for x in grbs_clean if x['PTFname'] == 'SN2009nz']
        y1 = [x[OHscale][0] for x in grbs_clean if x['PTFname'] == 'SN2009nz']
        plt.arrow(x1[0], y1[0], 0, -0.28, color=sncolors['GRB'], head_width=0.1,
                  head_length=0.1)

    temp1 = [float(x['logMs']) for x in grbs_clean if x['region'] == 'SNsite']
    temp2 = [x[OHscale][0] for x in grbs_clean if x['region'] == 'SNsite']
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in grbs_clean
             if x['region'] == 'SNsite']
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in grbs_clean
             if x['region'] == 'SNsite']
    temp5 = [float(x['elogMs']) for x in grbs_clean if x['region'] == 'SNsite']
    temp6 = [float(x['elogMs']) for x in grbs_clean if x['region'] == 'SNsite']
    temp10 = np.array([x['upperlimit'] for x in grbs_clean
                       if x['region'] == 'SNsite'], dtype=bool)
    point6 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                          fmt='D', color=sncolors['GRB'], markeredgecolor='none', label='SN-GRB')
    # plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
    #uplims=temp10, fmt='D', color=sncolors['GRB'], markeredgecolor='none', label='SN-GRB')
    # the uplims doesnt work, use arrow instead
    if True in temp10:
        x1 = [float(x['logMs']) for x in grbs_clean if x['PTFname'] == 'SN2013dx']
        y1 = [x[OHscale][0] for x in grbs_clean if x['PTFname'] == 'SN2013dx']
        plt.arrow(x1[0], y1[0], 0, -0.28, color=sncolors['GRB'], head_width=0.1, head_length=0.1)

    temp1 = [float(x['logMs']) for x in grbs_clean if x['region'] == 'SNsite'
             and x['PTFname'] != 'SN2013dx' and x['PTFname'] != 'SN2009nz']
    temp2 = [x[OHscale][0] for x in grbs_clean if x['region'] == 'SNsite'
             and x['PTFname'] != 'SN2013dx' and x['PTFname'] != 'SN2009nz']
    xavgs, yavgs = avgtrend3(temp1, temp2)
    plt.plot(xavgs, yavgs, color=sncolors['GRB'], linestyle='solid', linewidth=2)
    #if OHscale == 'PP04_O3N2':
    lcb, ucb, xavgs = avgtrend4(temp1, temp2)
    plt.fill_between(xavgs, lcb, ucb, color=sncolors['GRB'], alpha=0.2)

    #now also plot the SLSNe
    #want to plot metallicity against log(galaxy_mass), just data+errors, no fits
    
    point_slsn = plt.errorbar(slsn_t4['logM*'], slsn_z_dat[OHscale], xerr=[slsn_t4['e_logM*'],slsn_t4['E_logM*']], yerr=[slsn_z_dat['ne_'+OHscale], slsn_z_dat['pe_'+OHscale]], color = sncolors['slsn'], marker = '*', linestyle='None', label='SLSN Ic')

    #for kk,item in enumerate(grbs_clean):
    #       tempname = item['PTFname']
    #       tempname = tempname[-4:]
    #       tempx = float(item['logMs'])-0.45
    #       tempy = float(item[OHscale][0])+0.02
    #       plt.annotate(tempname, xy=(tempx, tempy), color='red', fontsize=fs-12)

    #if OHscale in gzk1[0].keys():
    #       temp1 = [x["LgMs"][0] for x in gzk2]
    #       temp2 = [x[OHscale][0] for x in gzk1]
    #       temp3 = [x[OHscale][0]-x[OHscale][1] for x in gzk1]
    #       temp4 = [x[OHscale][2]-x[OHscale][0] for x in gzk1]
    #       temp5 = [x["LgMs"][0]-x["LgMs"][3] for x in gzk2]
    #       temp6 = [x["LgMs"][4]-x["LgMs"][0] for x in gzk2]
    #       plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
    #fmt='g^', markeredgecolor='none')
    #       plt.annotate('12gzk', xy=(temp1[0]-0.4, temp2[0]+0.1), color='green')

    temp1 = [x["logMs"][0] for x in clean1 if x["SNtype"] == "uncertain"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "uncertain"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if
             x["SNtype"] == "uncertain"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if
             x["SNtype"] == "uncertain"]
    temp5 = [x["logMs"][0]-x["logMs"][3] for x in clean1 if
             x["SNtype"] == "uncertain"]
    temp6 = [x["logMs"][4]-x["logMs"][0] for x in clean1 if
             x["SNtype"] == "uncertain"]
    point7 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                          fmt='g^', fillstyle='none',
                          label='uncertain/weird SN subtype')

    #if i == 1:
    #       legend = ax.legend(loc='upper left', fontsize=fs-9, frameon=False,
    #numpoints=1)

    if i == 0:
        first_legend = ax.legend(handles=[point1, point2, point3, point4],
                                 loc='upper left', fontsize=fs-9,
                                 frameon=False, numpoints=1, handletextpad=0.1)
    if i == 1:
        second_legend = ax.legend(handles=[point7, point5, point6, point_slsn],
                                  loc='upper left', fontsize=fs-9,
                                  frameon=False, numpoints=1, handletextpad=0.1)
    #if i == 2:
    #       thrid_legend = ax.legend(handles=[], loc='upper left', fontsize=fs-9,
    #frameon=False, numpoints=1)
    #if i == 3:
    #       forth_legend = ax.legend(handles=[point7], loc='upper left',
    #fontsize=fs-9, frameon=False, numpoints=1)

    #if OHscale == 'KD02comb':
    #       temp1 = np.linspace(8.5, 11, 50)
    #       temp2 = 28.0974 - 7.23631*temp1 + 0.850344*temp1**2. - 0.0318315*temp1**3
    #       plt.plot(temp1, temp2, color='grey', linewidth=1.5)

    if OHscale == 'PP04_O3N2':
        #temp1 = np.linspace(8.5, 11, 50)
        #temp2 = 32.1488 -8.51258*temp1 + 0.976384*temp1**2. - 0.0359763*temp1**3
        #line1, = plt.plot(temp1, temp2, '--', color='grey', linewidth=1.5,
        #label='SDSS')
        line1, = plt.plot(sdss['PP04_x'], sdss['PP04_y'], '-.', linewidth=1.5,
                          color=sncolors['sdss'], label='SDSS')
        plt.fill_between(sdss['PP04_ux'], sdss['PP04_ly'], sdss['PP04_uy'],
                         color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'D13_N2S2_O3S2':
        line1, = plt.plot(sdss['D13_x'], sdss['D13_y'], '-.', linewidth=1.5,
                          color=sncolors['sdss'], label='SDSS')
        plt.fill_between(sdss['D13_ux'], sdss['D13_ly'], sdss['D13_uy'],
                         color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'KD02comb':
        line1, = plt.plot(sdss['KD02_x'], sdss['KD02_y'], '-.', linewidth=1.5,
                          color=sncolors['sdss'], label='SDSS')
        plt.fill_between(sdss['KD02_ux'], sdss['KD02_ly'], sdss['KD02_uy'],
                         color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'M08_N2Ha':
        line1, = plt.plot(sdss['M08_x'], sdss['M08_y'], '-.', linewidth=1.5,
                          color=sncolors['sdss'], label='SDSS')
        plt.fill_between(sdss['M08_ux'], sdss['M08_ly'], sdss['M08_uy'],
                         color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'M13_O3N2':
        line1, = plt.plot(sdss['M13_x'], sdss['M13_y'], '-.', linewidth=2.5,
                          color=sncolors['sdss'], label='SDSS')
        plt.fill_between(sdss['M13_ux'], sdss['M13_ly'], sdss['M13_uy'],
                         color=sncolors['sdss'], alpha=0.2)

    if i == 2:
        thrid_legend = ax.legend(handles=[point0, line1], loc='upper left',
                                 fontsize=fs-9, frameon=False, numpoints=1,
                                 handletextpad=0.1)


outfile = outpdf+'1.pdf'
plt.savefig(outfile, format='pdf')
plt.close(fig)

fig = plt.figure(figsize=(15, 12.5))
gspec = gridspec.GridSpec(4, 4)
gspec.update(wspace=0, hspace=0)

top_histogram = plt.subplot(gspec[0, 0:3])
side_histogram = plt.subplot(gspec[1:, 3])
lower_left = plt.subplot(gspec[1:, 0:3])

x_range = np.array([5., 12.])
y_range = (-3., 2.)

#ax = fig.add_subplot(111)
#fig.subplots_adjust(left=0.1, bottom=0.18)
fs = 20
ls = 20

plt.xlim(x_range[0], x_range[1])
plt.ylim(y_range[0], y_range[1])
plt.xlabel(r'$\log\ M_* {\rm\ [ M_{\odot}]}$', fontsize=fs)
plt.ylabel(r'$\log\ {\rm SFR\ [M_{\odot}yr^{-1}]}$', fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=ls)

temp7 = [np.log10(x[11]) for x in lvls]
temp8 = [np.log10(x[9]) for x in lvls]
temp9 = [x[9]*50 for x in lvls]
point0 = lower_left.scatter(temp7, temp8, s=temp9, color='lightgrey', label='LVL')
#xavgs, yavgs = avgtrend(temp7, temp8, smooth=1)

# read in sfsq trends for both SDSS and LVL galaxies
sfsq = {}
file5 = open(insfsq, 'r')
lines = file5.readlines()
flag = 0
for line in lines:
    if re.search(r'[a-zA-Z]+_[a-zA-Z]+', line):
        key = line[:-1]
        flag = 0
    if line.startswith('['):
        line = line[1:-2]
        vals = line.split(r', ')
        vals = [float(val) for val in vals]
        flag = 1
    if flag:
        sfsq[key] = vals
file5.close()

line0, = lower_left.plot(sfsq['lvl_x'], sfsq['lvl_y'], '--', linewidth=2.5,
                         color='lightgrey')
plt.fill_between(sfsq['lvl_ux'], sfsq['lvl_ly'], sfsq['lvl_uy'], color='lightgrey',
                 alpha=0.2)
line1, = lower_left.plot(sfsq['sdss_x'], sfsq['sdss_y'], '-.', linewidth=2.5,
                         color=sncolors['sdss'], label='SDSS')
plt.fill_between(sfsq['sdss_ux'], sfsq['sdss_ly'], sfsq['sdss_uy'], color=sncolors['sdss'],
                 alpha=0.2)


rr=32
lower_left.plot(x_range, x_range-8, color='grey')
lower_left.plot(x_range, x_range-9, color='grey')
lower_left.plot(x_range, x_range-10, color='grey')
lower_left.plot(x_range, x_range-11, color='grey')
lower_left.text(x_range[0], x_range[0]-6.8, 'log sSFR = $-$8 [yr$^{-1}$]', rotation=rr)
lower_left.text(x_range[0]+1, x_range[0]-7.3, '$-$9 [yr$^{-1}$]', rotation=rr)
lower_left.text(x_range[0]+2, x_range[0]-7.3, '$-$10 [yr$^{-1}$]', rotation=rr)
lower_left.text(x_range[0]+3, x_range[0]-7.3, '$-$11 [yr$^{-1}$]', rotation=rr)

temp1 = [x["logMs"][0] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
temp2 = [x["logSFR"][0] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
temp3 = [x["logSFR"][0]-x["logSFR"][3] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
temp4 = [x["logSFR"][4]-x["logSFR"][0] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
temp5 = [x["logMs"][0]-x["logMs"][3] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
temp6 = [x["logMs"][4]-x["logMs"][0] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
point1 = lower_left.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                             fmt='ks', fillstyle='none',
                             label='SN Ic (optical only)', mew=1.5, ms=10)

temp1 = [x["logMs"][0] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
temp2 = [x["logSFR"][0] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
temp3 = [x["logSFR"][0]-x["logSFR"][3] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
temp4 = [x["logSFR"][4]-x["logSFR"][0] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
temp5 = [x["logMs"][0]-x["logMs"][3] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
temp6 = [x["logMs"][4]-x["logMs"][0] for x in clean if x["SNtype"] == "Ic"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
point2 = lower_left.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                             fmt='ks', markeredgecolor='none',
                             label='SN Ic (optical+UV)', mew=1.5, ms=10)
xavgs, yavgs = avgtrend3(temp1, temp2)
lower_left.plot(xavgs, yavgs, color='black', linestyle='dotted', linewidth=2.5)

temp1 = [x["logMs"][0] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
temp2 = [x["logSFR"][0] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
temp3 = [x["logSFR"][0]-x["logSFR"][3] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
temp4 = [x["logSFR"][4]-x["logSFR"][0] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
temp5 = [x["logMs"][0]-x["logMs"][3] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
temp6 = [x["logMs"][4]-x["logMs"][0] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
point3 = lower_left.errorbar(temp1, temp2, xerr=[temp5, temp6],
                             yerr=[temp3, temp4], fmt = 'o', color=sncolors['Ic-BL'],
                             fillstyle='none', label='SN Ic-BL (optical only)',
                             mew=1.5, ms=10)

temp1 = [x["logMs"][0] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
temp2 = [x["logSFR"][0] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
temp3 = [x["logSFR"][0]-x["logSFR"][3] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
temp4 = [x["logSFR"][4]-x["logSFR"][0] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
temp5 = [x["logMs"][0]-x["logMs"][3] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
temp6 = [x["logMs"][4]-x["logMs"][0] for x in clean if x["SNtype"] == "Ic-BL"
         and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
point4 = lower_left.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                             fmt = 'o', color=sncolors['Ic-BL'], markeredgecolor='none',
                             label='SN Ic-BL (optical+UV)', mew=1.5, ms=10)
xavgs, yavgs = avgtrend3(temp1, temp2)
lower_left.plot(xavgs, yavgs, color=sncolors['Ic-BL'], linestyle='dashed', linewidth=2)

temp1 = [float(x["logMs"]) for x in grbs1 if x["region"] != "SNsite"]
temp2 = [np.log10(float(x["SFR"])) for x in grbs1 if x["region"] != "SNsite"]
temp3 = [float(x["elogSFR"]) for x in grbs1 if x["region"] != "SNsite"]
temp4 = [float(x["elogSFR"]) for x in grbs1 if x["region"] != "SNsite"]
temp5 = [float(x["elogMs"]) for x in grbs1 if x["region"] != "SNsite"]
temp6 = [float(x["elogMs"]) for x in grbs1 if x["region"] != "SNsite"]
point5 = lower_left.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                             fmt='D', color=sncolors['GRB'], label='SN-GRB (wAGN)', fillstyle='none',
                             ms=10, mew=1.5)

temp1 = [float(x["logMs"]) for x in grbs1 if x["region"] == "SNsite"]
temp2 = [np.log10(float(x["SFR"])) for x in grbs1 if x["region"] == "SNsite"]
temp3 = [float(x["elogSFR"]) for x in grbs1 if x["region"] == "SNsite"]
temp4 = [float(x["elogSFR"]) for x in grbs1 if x["region"] == "SNsite"]
temp5 = [float(x["elogMs"]) for x in grbs1 if x["region"] == "SNsite"]
temp6 = [float(x["elogMs"]) for x in grbs1 if x["region"] == "SNsite"]
point6 = lower_left.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                             fmt='D', color=sncolors['GRB'], label='SN-GRB', markeredgecolor='none',
                             ms=10, mew=1.5)

xavgs, yavgs = avgtrend3(temp1, temp2)
lower_left.plot(xavgs, yavgs, color=sncolors['GRB'], linestyle='solid', linewidth=2)
lcb, ucb, xavgs = avgtrend4(temp1, temp2)
plt.fill_between(xavgs, lcb, ucb, color=sncolors['GRB'], alpha=0.2)

#temp1 = [x["LgMs"][0] for x in gzk2]
#temp2 = [x["LgSFR"][0] for x in gzk2]
#temp3 = [x["LgSFR"][0]-x["LgSFR"][3] for x in gzk2]
#temp4 = [x["LgSFR"][4]-x["LgSFR"][0] for x in gzk2]
#temp5 = [x["LgMs"][0]-x["LgMs"][3] for x in gzk2]
#temp6 = [x["LgMs"][4]-x["LgMs"][0] for x in gzk2]
#lower_left.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4], fmt='g^',
#markeredgecolor='none', ms=10, mew=1.5)
#lower_left.annotate('12gzk', xy=(temp1[0]+0.1, temp2[0]-0.1), color='green')

temp1 = [x["logMs"][0] for x in clean if x["SNtype"] == "uncertain"]
temp2 = [x["logSFR"][0] for x in clean if x["SNtype"] == "uncertain"]
temp3 = [x["logSFR"][0]-x["logSFR"][3] for x in clean if x["SNtype"] == "uncertain"]
temp4 = [x["logSFR"][4]-x["logSFR"][0] for x in clean if x["SNtype"] == "uncertain"]
temp5 = [x["logMs"][0]-x["logMs"][3] for x in clean if x["SNtype"] == "uncertain"]
temp6 = [x["logMs"][4]-x["logMs"][0] for x in clean if x["SNtype"] == "uncertain"]
point7 = lower_left.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                             fmt='g^', fillstyle='none',
                             label='uncertain/weird SN subtype', mew=1.5, ms=10)

yerr_slsn = [0.434*slsn_t4['e_SFR']/slsn_t4['SFR'],0.434*slsn_t4['E_SFR']/slsn_t4['SFR']]
point_slsn = lower_left.errorbar(slsn_t4['logM*'], np.log10(slsn_t4['SFR']), xerr=[slsn_t4['e_logM*'],slsn_t4['E_logM*']], yerr=yerr_slsn, color = sncolors['slsn'], marker = '*', linestyle='None', label='SLSN Ic')

first_legend = lower_left.legend(handles=[point1, point2, point3, point4,
                                          point7, point5, point6, point0, line1, point_slsn],
                                 loc='upper left', fontsize=fs-9, frameon=False,
                                 numpoints=1,
                                 handletextpad=0.01)

#second_legend = lower_left.legend(handles=[point0, line1], loc='lower right',
#fontsize=fs-9, frameon=False, numpoints=1)

# print min([x["logMs"][0] for x in clean]), max([x["logMs"][0] for x in clean])
# print min([x["logSFR"][0] for x in clean]), max([x["logSFR"][0] for x in clean])

#xavgs, yavgs = avgtrend2(temp7, temp8)
#xavgs, yavgs = avgtrend3(temp7, temp8)
#lower_left.plot(xavgs, yavgs, color='lightgrey', linestyle='dashdotted', linewidth=2.5)

temp0 = [x["logMs"][0] for x in clean if x["SNtype"] == "Ic"]
nx, xbins, ptchs = top_histogram.hist(temp0, bins=1000, normed=1, cumulative=True)
xtemp1 = [xbins[0]]
xtemp1.extend(xbins[:-1])
ytemp1 = [0]
ytemp1.extend(nx)

temp0 = [x["logMs"][0] for x in clean if x["SNtype"] == "Ic-BL"]
nx, xbins, ptchs = top_histogram.hist(temp0, bins=1000, normed=1, cumulative=True)
xtemp2 = [xbins[0]]
xtemp2.extend(xbins[:-1])
ytemp2 = [0]
ytemp2.extend(nx)

temp0 = [float(x["logMs"]) for x in grbs1]
nx, xbins, ptchs = top_histogram.hist(temp0, bins=1000, normed=1, cumulative=True)
xtemp3 = [xbins[0]]
xtemp3.extend(xbins[:-1])
ytemp3 = [0]
ytemp3.extend(nx)

temp0 = slsn_t4['logM*']
nx, xbins, ptchs = top_histogram.hist(temp0, bins=1000, normed=1, cumulative=True)
xtemp4 = [xbins[0]]
xtemp4.extend(xbins[:-1])
ytemp4 = [0]
ytemp4.extend(nx)

top_histogram.clear()

top_histogram.plot(xtemp1, ytemp1, color='black', linestyle='dotted', linewidth=2.5)
top_histogram.plot(xtemp2, ytemp2, color=sncolors['Ic-BL'], linestyle='dashed', linewidth=2.5)
top_histogram.plot(xtemp3, ytemp3, color=sncolors['GRB'], linestyle='solid', linewidth=2.5)
top_histogram.plot(xtemp4, ytemp4, color=sncolors['slsn'], linestyle='solid', linewidth=2.5)

top_histogram.set_xticklabels([])
top_histogram.set_yticklabels([0.5, 1])
top_histogram.set_yticks([0.5, 1])
plt.setp(top_histogram.get_yticklabels(), fontsize=fs-5)

top_histogram.set_ylim(0, 1)

temp0 = [x["logSFR"][0] for x in clean if x["SNtype"] == "Ic"]
nx, xbins, ptchs = side_histogram.hist(temp0, bins=1000, normed=1, cumulative=True,
                                       orientation='horizontal')
xtemp1 = [xbins[0]]
xtemp1.extend(xbins[:-1])
ytemp1 = [0]
ytemp1.extend(nx)

temp0 = [x["logSFR"][0] for x in clean if x["SNtype"] == "Ic-BL"]
nx, xbins, ptchs = side_histogram.hist(temp0, bins=1000, normed=1, cumulative=True,
                                       orientation='horizontal')
xtemp2 = [xbins[0]]
xtemp2.extend(xbins[:-1])
ytemp2 = [0]
ytemp2.extend(nx)

temp0 = [np.log10(float(x["SFR"])) for x in grbs1]
nx, xbins, ptchs = side_histogram.hist(temp0, bins=1000, normed=1, cumulative=True,
                                       orientation='horizontal')
xtemp3 = [xbins[0]]
xtemp3.extend(xbins[:-1])
ytemp3 = [0]
ytemp3.extend(nx)

temp0 = np.log10(slsn_t4['SFR'])
nx, xbins, ptchs = side_histogram.hist(temp0, bins=1000, normed=1, cumulative=True,
                                       orientation='horizontal')
xtemp4 = [xbins[0]]
xtemp4.extend(xbins[:-1])
ytemp4 = [0]
ytemp4.extend(nx)

side_histogram.clear()

side_histogram.plot(ytemp1, xtemp1, color='black', linestyle='dotted', linewidth=2.5)
side_histogram.plot(ytemp2, xtemp2, color=sncolors['Ic-BL'], linestyle='dashed', linewidth=2.5)
side_histogram.plot(ytemp3, xtemp3, color=sncolors['GRB'], linestyle='solid', linewidth=2.5)
side_histogram.plot(ytemp4, xtemp4, color=sncolors['slsn'], linestyle='solid', linewidth=2.5)

side_histogram.set_yticklabels([])
side_histogram.set_xticklabels([0.5, 1])
side_histogram.set_xticks([0.5, 1])
plt.setp(side_histogram.get_xticklabels(), fontsize=fs-5)

side_histogram.set_xlim(0, 1)

for ax in [top_histogram, lower_left]:
    ax.set_xlim(x_range[0], x_range[1])
for ax in [side_histogram, lower_left]:
    ax.set_ylim(y_range[0], y_range[1])

for ax in [lower_left, top_histogram, side_histogram]:
    ax.minorticks_on()
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 7)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 1)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 7)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 1)
    plt.setp(ax.yaxis.get_ticklines(minor=True), 'markersize', 4)
    plt.setp(ax.yaxis.get_ticklines(minor=True), 'markeredgewidth', 1)
    plt.setp(ax.xaxis.get_ticklines(minor=True), 'markersize', 4)
    plt.setp(ax.xaxis.get_ticklines(minor=True), 'markeredgewidth', 1)

# plt.show()
outfile = outpdf+'2.pdf'
plt.savefig(outfile, format='pdf')
plt.close(fig)

x_range = (-12.4, -6.8)
y_range = (7.3, 9.6)

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(left=0.1, bottom=0.18, hspace=0, wspace=0)
fs = 20
ls = 15

# read in ohsq trends for both SDSS and LVL galaxies
ohsq = {}
file6 = open(inohsq, 'r')
lines = file6.readlines()
flag = 0
for line in lines:
    if re.search(r'[a-zA-Z0-9]+_[a-zA-Z0-9]+', line):
        key = line[:-1]
        flag = 0
    if line.startswith('['):
        line = line[1:-2]
        vals = line.split(r', ')
        vals = [float(val) for val in vals]
        flag = 1
    if flag:
        ohsq[key] = vals
file6.close()

#print ohsq.keys()

OHscales = ['KD02comb', 'PP04_O3N2', 'M08_N2Ha', 'M13_O3N2']

for i, OHscale in enumerate(OHscales):
    ax = fig.add_subplot(221+i)

    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.minorticks_on()
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 7)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 1)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 7)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 1)
    plt.setp(ax.yaxis.get_ticklines(minor=True), 'markersize', 4)
    plt.setp(ax.yaxis.get_ticklines(minor=True), 'markeredgewidth', 1)
    plt.setp(ax.xaxis.get_ticklines(minor=True), 'markersize', 4)
    plt.setp(ax.xaxis.get_ticklines(minor=True), 'markeredgewidth', 1)
    plt.tick_params(axis='both', which='major', labelsize=ls)
    if i < 2:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        #plt.xlabel('Host Galaxy M_B (mag)', fontsize=fs)
        plt.xlabel(r'$\log\ {\rm sSFR\ [yr^{-1}]}$', fontsize=fs)
    if i % 2 == 1:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
    else:
        plt.ylabel(r'$\log\ {\rm(O/H)+12}$', fontsize=fs)
    plt.annotate(OHscale, xy=(0.1, 0.1), xycoords='axes fraction', fontsize=fs-7)

    if OHscale == 'PP04_O3N2':
        #line0, = plt.plot(ohsq['lvl_x'], ohsq['lvl_y'], ':', linewidth=2.5,
        #color='lightgrey')
        #plt.fill_between(ohsq['lvl_ux'], ohsq['lvl_ly'], ohsq['lvl_uy'],
        #color='lightgrey', alpha=0.2)
        line1, = plt.plot(ohsq['PP04_x'], ohsq['PP04_y'], '-.', linewidth=2.5,
                         color=sncolors['sdss'], label='SDSS')
        plt.fill_between(ohsq['PP04_ux'], ohsq['PP04_ly'], ohsq['PP04_uy'],
                         color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'M13_O3N2':
        line1, = plt.plot(ohsq['M13_x'], ohsq['M13_y'], '-.', linewidth=2.5,
                         color=sncolors['sdss'], label='SDSS')
        plt.fill_between(ohsq['M13_ux'], ohsq['M13_ly'], ohsq['M13_uy'],
                         color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'KD02comb':
        line1, = plt.plot(ohsq['KD02_x'], ohsq['KD02_y'], '-.', linewidth=2.5,
                         color=sncolors['sdss'], label='SDSS')
        plt.fill_between(ohsq['KD02_ux'], ohsq['KD02_ly'], ohsq['KD02_uy'],
                         color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'M08_N2Ha':
        line1, = plt.plot(ohsq['M08_x'], ohsq['M08_y'], '-.', linewidth=2.5,
                          color=sncolors['sdss'], label='SDSS')
        plt.fill_between(ohsq['M08_ux'], ohsq['M08_ly'], ohsq['M08_uy'],
                         color=sncolors['sdss'], alpha=0.2)

    temp7 = [np.log10(x[9])-np.log10(x[11]) for x in lvls]
    temp8 = [x[12] for x in lvls]
    temp9 = [x[9]*50 for x in lvls]
    point0 = plt.scatter(temp7, temp8, s=temp9, color='lightgrey')

    clean1 = []
    for cleanone in clean:
        if OHscale in cleanone.keys():
            clean1.append(cleanone)

    temp1 = [x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic" and
             len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic" and
             len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic" and
             len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic" and
             len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp5 = [x["logsSFR"][0]-x["logsSFR"][3] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp6 = [x["logsSFR"][4]-x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    point1 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                          fmt='ks', fillstyle='none', label='SN Ic (optical only)')

    temp1 = [x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp5 = [x["logsSFR"][0]-x["logsSFR"][3] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp6 = [x["logsSFR"][4]-x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    point2 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                          fmt='ks', markeredgecolor='none',
                          label='SN Ic (optical+UV)')
    #xavgs, yavgs = avgtrend3(temp1, temp2)
    #plt.plot(xavgs, yavgs, color='black', linestyle='solid', linewidth=2)

    temp1 = [x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp5 = [x["logsSFR"][0]-x["logsSFR"][3] for x in clean1 if
             x["SNtype"] == "Ic-BL" and len(x["modelmag"]) != 7]
    #x["region"] != "SNsite"]
    temp6 = [x["logsSFR"][4]-x["logsSFR"][0] for x in clean1 if
             x["SNtype"] == "Ic-BL" and len(x["modelmag"]) != 7]
    #x["region"] != "SNsite"]
    point3 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                          fmt = 'o', color=sncolors['Ic-BL'], fillstyle='none',
                          label='SN Ic-BL (optical only)')

    temp1 = [x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp5 = [x["logsSFR"][0]-x["logsSFR"][3] for x in clean1
             if x["SNtype"] == "Ic-BL" and len(x["modelmag"]) == 7]
    #x["region"] == "SNsite"]
    temp6 = [x["logsSFR"][4]-x["logsSFR"][0] for x in clean1 if
             x["SNtype"] == "Ic-BL" and len(x["modelmag"]) == 7]
    #x["region"] == "SNsite"]
    point4 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6],
                          yerr=[temp3, temp4], fmt = 'o', color=sncolors['Ic-BL'], markeredgecolor='none',
                          label='SN Ic-BL (optical+UV)')
    # print min([x["logsSFR"][0] for x in clean]), max([x["logsSFR"][0]
    #for x in clean])
    # print min([x[OHscale][0] for x in clean]), max([x[OHscale][0] for x in clean])
    #xavgs, yavgs = avgtrend3(temp1, temp2)
    #plt.plot(xavgs, yavgs, color='blue', linestyle='dashed', linewidth=2)


    grbs_clean = []
    for grb1 in grbs1:
        if OHscale in grb1.keys():
            if grb1["PTFname"] == "SN2013dx":
                grb1[OHscale][1] = grb1[OHscale][0]
                grb1[OHscale][2] = grb1[OHscale][0]
                grb1["upperlimit"] = 1
            elif grb1["PTFname"] == "SN2009nz":
                if OHscale in ['PP04_O3N2', 'M08_N2Ha']:
                    grb1[OHscale][1] = grb1[OHscale][0]
                    grb1[OHscale][2] = grb1[OHscale][0]
                    grb1["upperlimit"] = 1
                else:
                    grb1["upperlimit"] = 0
            else:
                grb1["upperlimit"] = 0
            grbs_clean.append(grb1)

    temp1 = [np.log10(float(x["SFR"]))-float(x["logMs"]) for x in grbs_clean if
             x['region'] != 'SNsite']
    temp2 = [x[OHscale][0] for x in grbs_clean if x['region'] != 'SNsite']
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in grbs_clean if
             x['region'] != 'SNsite']
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in grbs_clean if
             x['region'] != 'SNsite']
    temp5 = [np.sqrt((float(x["elogMs"]))**2.+(float(x["elogSFR"]))**2.)
             for x in grbs_clean if x['region'] != 'SNsite']
    temp10 = np.array([x['upperlimit'] for x in grbs_clean
                       if x['region'] != 'SNsite'], dtype=bool)
    point5 = plt.errorbar(temp1, temp2, xerr=temp5, yerr=[temp3, temp4],
                          fmt='D', color=sncolors['GRB'], fillstyle='none', label='SN-GRB (wAGN)')
    if True in temp10:
        x1 = [np.log10(float(x["SFR"]))-float(x["logMs"]) for x in
              grbs_clean if x['PTFname'] == 'SN2009nz']
        y1 = [x[OHscale][0] for x in grbs_clean if x['PTFname'] == 'SN2009nz']
        plt.arrow(x1[0], y1[0], 0, -0.28, color=sncolors['GRB'], head_width=0.1,
                  head_length=0.1)

    temp1 = [np.log10(float(x["SFR"]))-float(x["logMs"])
             for x in grbs_clean if x['region'] == 'SNsite']
    temp2 = [x[OHscale][0] for x in grbs_clean if x['region'] == 'SNsite']
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in grbs_clean if
             x['region'] == 'SNsite']
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in grbs_clean if
             x['region'] == 'SNsite']
    temp5 = [np.sqrt((float(x["elogMs"]))**2.+(float(x["elogSFR"]))**2.)
             for x in grbs_clean if x['region'] == 'SNsite']
    temp10 = np.array([x['upperlimit'] for x in grbs_clean if
                       x['region'] == 'SNsite'], dtype=bool)
    point6 = plt.errorbar(temp1, temp2, xerr=temp5, yerr=[temp3, temp4],
                          fmt='D', color=sncolors['GRB'], markeredgecolor='none', label='SN-GRB')
    # the uplims doesnt work, use arrow instead
    if True in temp10:
        x1 = [np.log10(float(x["SFR"]))-float(x["logMs"])
              for x in grbs_clean if x['PTFname'] == 'SN2013dx']
        y1 = [x[OHscale][0] for x in grbs_clean if x['PTFname'] == 'SN2013dx']
        plt.arrow(x1[0], y1[0], 0, -0.28, color=sncolors['GRB'],
                  head_width=0.1, head_length=0.1)


    tempx = zip(temp1, temp10)
    tempy = zip(temp2, temp10)
    temp1 = [x[0] for x in tempx if x[1] == False]
    temp2 = [x[0] for x in tempy if x[1] == False]
    xavgs, yavgs = avgtrend3(temp1, temp2)
    plt.plot(xavgs, yavgs, color=sncolors['GRB'], linestyle='solid', linewidth=2)
    #if OHscale == 'PP04_O3N2':
    lcb, ucb, xavgs = avgtrend4(temp1, temp2)
    plt.fill_between(xavgs, lcb, ucb, color=sncolors['GRB'], alpha=0.2)


    #if i == 2:
    #       legend = plt.legend(loc='upper right', fontsize=fs-9,
    #frameon=False, numpoints=1)

    #if OHscale in gzk1[0].keys():
    #       temp1 = [x["LgSFR_spec"][0] for x in gzk2]
    #       temp2 = [x[OHscale][0] for x in gzk1]
    #       temp3 = [x[OHscale][0]-x[OHscale][1] for x in gzk1]
    #       temp4 = [x[OHscale][2]-x[OHscale][0] for x in gzk1]
    #       temp5 = [x["LgSFR_spec"][0]-x["LgSFR_spec"][3] for x in gzk2]
    #       temp6 = [x["LgSFR_spec"][4]-x["LgSFR_spec"][0] for x in gzk2]
    #       plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4], fmt='g^', markeredgecolor='none')
    #       plt.annotate('12gzk', xy=(temp1[0]+0.1, temp2[0]-0.1), color='green')

    temp1 = [x["logsSFR"][0] for x in clean1 if x["SNtype"] == "uncertain"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "uncertain"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if
             x["SNtype"] == "uncertain"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if
             x["SNtype"] == "uncertain"]
    temp5 = [x["logsSFR"][0]-x["logsSFR"][3] for x in clean1 if
             x["SNtype"] == "uncertain"]
    temp6 = [x["logsSFR"][4]-x["logsSFR"][0] for x in clean1 if
             x["SNtype"] == "uncertain"]
    point7 = plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4],
                          fmt='g^', fillstyle='none', label='uncertain/weird')

    xerr_1 = []
    xerr_2 = []
    #calculate errorbars for log10(sSFR)
    '''
    for k in range(0, len(slsn_t4['l_SFR'])):
        if slsn_t4['l_SFR'][k] == '<':
            xerr_1.append(0)
            xerr_2.append(0)
        else:
            xerr_1.append(np.sqrt((0.434*slsn_t4['e_SFR'][k]/slsn_t4['SFR'][k])**2+slsn_t4['e_logM*'][k]**2))
            xerr_2.append(np.sqrt((0.434*slsn_t4['E_SFR'][k]/slsn_t4['SFR'][k])**2+slsn_t4['E_logM*'][k]**2))
    '''
    t5_orig = ascii.read(inslsn5_orig)
    index_lha = np.where(t5_orig['l_FHa'] == '<')
    
    names_lha = t5_orig['PTF'][index_lha]
    
    m=0
    for ptfname in slsn_t4['PTF']:
        if ptfname in names_lha:
            xerr_1 =np.sqrt((0.434*slsn_t4['e_SFR']/slsn_t4['SFR'])**2+slsn_t4['e_logM*']**2)[m]
            xerr_2 =np.sqrt((0.434*slsn_t4['E_SFR']/slsn_t4['SFR'])**2+slsn_t4['E_logM*']**2)[m]
            xerr_8 = [[xerr_1], [xerr_2]]
            yerr_8=[[slsn_z_dat['ne_'+OHscale][m]], [slsn_z_dat['pe_'+OHscale][m]]]
            point8 = plt.errorbar(np.log10(slsn_t4['SFR'][m])-slsn_t4['logM*'][m], slsn_z_dat[OHscale][m], xerr=xerr_8, yerr=yerr_8, marker='*', color=sncolors['slsn'], label='SLSN Ic (Ha UL)', linestyle='None')

        else:
            xerr_1 =np.sqrt((0.434*slsn_t4['e_SFR']/slsn_t4['SFR'])**2+slsn_t4['e_logM*']**2)[m]
            xerr_2 =np.sqrt((0.434*slsn_t4['E_SFR']/slsn_t4['SFR'])**2+slsn_t4['E_logM*']**2)[m]
            xerr_8 = [[xerr_1], [xerr_2]]
            yerr_8=[[slsn_z_dat['ne_'+OHscale][m]], [slsn_z_dat['pe_'+OHscale][m]]]
            point9 = plt.errorbar(np.log10(slsn_t4['SFR'][m])-slsn_t4['logM*'][m], slsn_z_dat[OHscale][m], xerr=xerr_8, yerr=yerr_8, marker='o', mfc='none', color=sncolors['slsn'], label='SLSN Ic', linestyle='None')

        m+=1            
    '''
    
    index_ul = np.where(slsn_t4['l_SFR'] == '<')
    index_nul = np.where(slsn_t4['l_SFR'] != '<')
    xerr_1 =np.sqrt((0.434*slsn_t4['e_SFR']/slsn_t4['SFR'])**2+slsn_t4['e_logM*']**2)[index_nul]
    xerr_2 =np.sqrt((0.434*slsn_t4['E_SFR']/slsn_t4['SFR'])**2+slsn_t4['E_logM*']**2)[index_nul]
    xerr_8 = [xerr_1, xerr_2]
    point8 = plt.errorbar(np.log10(slsn_t4['SFR'][index_nul])-slsn_t4['logM*'][index_nul], slsn_z_dat[OHscale][index_nul], xerr=xerr_8, yerr=[slsn_z_dat['ne_'+OHscale][index_nul], slsn_z_dat['pe_'+OHscale][index_nul]], marker='*', color=sncolors['slsn'], label='SLSN Ic', linestyle='None')

    xerr_1 =np.sqrt((0.434*slsn_t4['e_SFR']/slsn_t4['SFR'])**2+slsn_t4['e_logM*']**2)[index_ul]
    xerr_2 =np.sqrt((0.434*slsn_t4['E_SFR']/slsn_t4['SFR'])**2+slsn_t4['E_logM*']**2)[index_ul]
    xerr_8 = [xerr_1, xerr_2]
    point9 = plt.errorbar(np.log10(slsn_t4['SFR'][index_ul])-slsn_t4['logM*'][index_ul], slsn_z_dat[OHscale][index_ul], xerr=xerr_8, yerr=[slsn_z_dat['ne_'+OHscale][index_ul], slsn_z_dat['pe_'+OHscale][index_ul]], marker='o', markersize = 20, color=sncolors['slsn'], label='SLSN Ic (UL)', linestyle='None')
    '''
    #print slsn_z_dat[OHscale][index_ul]
    
    #if i == 0:
    #       first_legend = plt.legend(handles=[point1, point2, point3, point4], loc='upper right', fontsize=fs-12, frameon=False, numpoints=1)
    #if i == 1:
    #       second_legend = plt.legend(handles=[point5, point6, point7], loc='upper right', fontsize=fs-12, frameon=False, numpoints=1)
    #if i == 2:
    #       thrid_legend = plt.legend(handles=[point0, line1], loc='upper right', fontsize=fs-12, frameon=False, numpoints=1)

#plt.legend()
# plt.show()
outfile = outpdf+'3.pdf'
plt.savefig(outfile, format='pdf')
plt.close(fig)

y_range = (-12.4, -6.8)
x_range = (7.3, 9.6)

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(left=0.1, bottom=0.18, hspace=0, wspace=0)
fs = 20
ls = 15

# read in ohsq trends for both SDSS and LVL galaxies
sSFRsq = {}
file7 = open(insSFRsq, 'r')
lines = file7.readlines()
flag = 0
for line in lines:
    if re.search(r'[a-zA-Z0-9]+_[a-zA-Z0-9]+', line):
        key = line[:-1]
        flag = 0
    if line.startswith('['):
        line = line[1:-2]
        vals = line.split(r', ')
        vals = [float(val) for val in vals]
        flag = 1
    if flag:
        sSFRsq[key] = vals
file7.close()

for i, OHscale in enumerate(OHscales):
    ax = fig.add_subplot(221+i)

    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.minorticks_on()
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 7)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 1)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 7)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 1)
    plt.setp(ax.yaxis.get_ticklines(minor=True), 'markersize', 4)
    plt.setp(ax.yaxis.get_ticklines(minor=True), 'markeredgewidth', 1)
    plt.setp(ax.xaxis.get_ticklines(minor=True), 'markersize', 4)
    plt.setp(ax.xaxis.get_ticklines(minor=True), 'markeredgewidth', 1)
    plt.tick_params(axis='both', which='major', labelsize=ls)
    if i < 2:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        #plt.xlabel('Host Galaxy M_B (mag)', fontsize=fs)
        plt.xlabel(r'$\log\ {\rm(O/H)+12}$', fontsize=fs)
    if i % 2 == 1:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
    else:
        plt.ylabel(r'$\log\ {\rm sSFR\ [yr^{-1}]}$', fontsize=fs)
    plt.annotate(OHscale, xy=(0.1, 0.1), xycoords='axes fraction', fontsize=fs-7)

    if OHscale == 'PP04_O3N2':
        #line0, = plt.plot(ohsq['lvl_x'], ohsq['lvl_y'], ':', linewidth=2.5, color='lightgrey')
        #plt.fill_between(ohsq['lvl_ux'], ohsq['lvl_ly'], ohsq['lvl_uy'], color='lightgrey', alpha=0.2)
        line1, = plt.plot(sSFRsq['PP04_x'], sSFRsq['PP04_y'], '-.',
                          linewidth=2.5, color=sncolors['sdss'], label='SDSS')
        plt.fill_between(sSFRsq['PP04_ux'], sSFRsq['PP04_ly'],
                         sSFRsq['PP04_uy'], color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'D13_N2S2_O3S2':
        line1, = plt.plot(sSFRsq['D13_x'], sSFRsq['D13_y'], '-.',
                          linewidth=2.5, color=sncolors['sdss'], label='SDSS')
        plt.fill_between(sSFRsq['D13_ux'], sSFRsq['D13_ly'], sSFRsq['D13_uy'],
                         color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'M13_O3N2':
        line1, = plt.plot(sSFRsq['M13_x'], sSFRsq['M13_y'], '-.', linewidth=2.5,
                          color=sncolors['sdss'], label='SDSS')
        plt.fill_between(sSFRsq['M13_ux'], sSFRsq['M13_ly'], sSFRsq['M13_uy'],
                         color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'KD02comb':
        line1, = plt.plot(sSFRsq['KD02_x'], sSFRsq['KD02_y'], '-.',
                          linewidth=2.5, color=sncolors['sdss'], label='SDSS')
        plt.fill_between(sSFRsq['KD02_ux'], sSFRsq['KD02_ly'], sSFRsq['KD02_uy'],
                         color=sncolors['sdss'], alpha=0.2)
    if OHscale == 'M08_N2Ha':
        line1, = plt.plot(sSFRsq['M08_x'], sSFRsq['M08_y'], '-.',
                          linewidth=2.5, color=sncolors['sdss'], label='SDSS')
        plt.fill_between(sSFRsq['M08_ux'], sSFRsq['M08_ly'], sSFRsq['M08_uy'],
                         color=sncolors['sdss'], alpha=0.2)

    temp7 = [np.log10(x[9])-np.log10(x[11]) for x in lvls]
    temp8 = [x[12] for x in lvls]
    temp9 = [x[9]*50 for x in lvls]
    point0 = plt.scatter(temp8, temp7, s=temp9, color='lightgrey')

    clean1 = []
    for cleanone in clean:
        if OHscale in cleanone.keys():
            clean1.append(cleanone)

    temp1 = [x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp5 = [x["logsSFR"][0]-x["logsSFR"][3] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp6 = [x["logsSFR"][4]-x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    point1 = plt.errorbar(temp2, temp1, xerr=[temp3, temp4], yerr=[temp5, temp6],
                          fmt='ks', fillstyle='none', label='SN Ic (optical only)')

    temp1 = [x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp5 = [x["logsSFR"][0]-x["logsSFR"][3] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp6 = [x["logsSFR"][4]-x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    point2 = plt.errorbar(temp2, temp1, xerr=[temp3, temp4], yerr=[temp5, temp6],
                          fmt='ks', markeredgecolor='none', label='SN Ic (optical+UV)')
    #xavgs, yavgs = avgtrend3(temp2, temp1)
    #plt.plot(xavgs, yavgs, color='black', linestyle='solid', linewidth=2)

    temp1 = [x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) != 7] #x["region"] != "SNsite"]
    temp5 = [x["logsSFR"][0]-x["logsSFR"][3] for x in clean1 if
             x["SNtype"] == "Ic-BL" and len(x["modelmag"]) != 7]
    #x["region"] != "SNsite"]
    temp6 = [x["logsSFR"][4]-x["logsSFR"][0] for x in clean1 if
             x["SNtype"] == "Ic-BL" and len(x["modelmag"]) != 7]
    #x["region"] != "SNsite"]
    point3 = plt.errorbar(temp2, temp1, xerr=[temp3, temp4], yerr=[temp5, temp6],
                          fmt = 'o', color=sncolors['Ic-BL'], fillstyle='none', label='SN Ic-BL (optical only)')

    temp1 = [x["logsSFR"][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1 if x["SNtype"] == "Ic-BL"
             and len(x["modelmag"]) == 7] #x["region"] == "SNsite"]
    temp5 = [x["logsSFR"][0]-x["logsSFR"][3] for x in clean1 if
             x["SNtype"] == "Ic-BL" and len(x["modelmag"]) == 7]
    #x["region"] == "SNsite"]
    temp6 = [x["logsSFR"][4]-x["logsSFR"][0] for x in clean1 if
             x["SNtype"] == "Ic-BL" and len(x["modelmag"]) == 7]
    #x["region"] == "SNsite"]
    point4 = plt.errorbar(temp2, temp1, xerr=[temp3, temp4], yerr=[temp5, temp6],
                          fmt = 'o', color=sncolors['Ic-BL'], markeredgecolor='none',
                          label='SN Ic-BL (optical+UV)')
    # print min([x["logsSFR"][0] for x in clean]), max([x["logsSFR"][0] for x in clean])
    # print min([x[OHscale][0] for x in clean]), max([x[OHscale][0] for x in clean])
    #xavgs, yavgs = avgtrend3(temp2, temp1)
    #plt.plot(xavgs, yavgs, color='blue', linestyle='dashed', linewidth=2)


    grbs_clean = []
    for grb1 in grbs1:
        if OHscale in grb1.keys():
            if grb1["PTFname"] == "SN2013dx":
                grb1[OHscale][1] = grb1[OHscale][0]
                grb1[OHscale][2] = grb1[OHscale][0]
                grb1["upperlimit"] = 1
            elif grb1["PTFname"] == "SN2009nz":
                if OHscale in ['PP04_O3N2', 'M08_N2Ha']:
                    grb1[OHscale][1] = grb1[OHscale][0]
                    grb1[OHscale][2] = grb1[OHscale][0]
                    grb1["upperlimit"] = 1
                else:
                    grb1["upperlimit"] = 0
            else:
                grb1["upperlimit"] = 0
            grbs_clean.append(grb1)

    temp1 = [np.log10(float(x["SFR"]))-float(x["logMs"]) for x in grbs_clean if
             x['region'] != 'SNsite']
    temp2 = [x[OHscale][0] for x in grbs_clean if x['region'] != 'SNsite']
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in grbs_clean if
             x['region'] != 'SNsite']
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in grbs_clean if
             x['region'] != 'SNsite']
    temp5 = [np.sqrt((float(x["elogMs"]))**2.+(float(x["elogSFR"]))**2.)
             for x in grbs_clean if x['region'] != 'SNsite']
    temp10 = np.array([x['upperlimit'] for x in grbs_clean if
                       x['region'] != 'SNsite'], dtype=bool)
    point5 = plt.errorbar(temp2, temp1, xerr=[temp3, temp4], yerr=temp5,
                          fmt='D', color=sncolors['GRB'], fillstyle='none', label='SN-GRB (wAGN)')
    if True in temp10:
        x1 = [np.log10(float(x["SFR"]))-float(x["logMs"])
              for x in grbs_clean if x['PTFname'] == 'SN2009nz']
        y1 = [x[OHscale][0] for x in grbs_clean if x['PTFname'] == 'SN2009nz']
        plt.arrow(y1[0], x1[0], -0.28, 0, color=sncolors['GRB'], head_width=0.1,
                  head_length=0.1)

    temp1 = [np.log10(float(x["SFR"]))-float(x["logMs"]) for x in grbs_clean
             if x['region'] == 'SNsite']
    temp2 = [x[OHscale][0] for x in grbs_clean if x['region'] == 'SNsite']
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in grbs_clean if
             x['region'] == 'SNsite']
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in grbs_clean if
             x['region'] == 'SNsite']
    temp5 = [np.sqrt((float(x["elogMs"]))**2.+(float(x["elogSFR"]))**2.)
             for x in grbs_clean if x['region'] == 'SNsite']
    temp10 = np.array([x['upperlimit'] for x in grbs_clean
                       if x['region'] == 'SNsite'], dtype=bool)
    point6 = plt.errorbar(temp2, temp1, xerr=[temp3, temp4], yerr=temp5,
                          fmt='D', color=sncolors['GRB'], markeredgecolor='none', label='SN-GRB')
    # the uplims doesnt work, use arrow instead
    if True in temp10:
        x1 = [np.log10(float(x["SFR"]))-float(x["logMs"])
              for x in grbs_clean if x['PTFname'] == 'SN2013dx']
        y1 = [x[OHscale][0] for x in grbs_clean if x['PTFname'] == 'SN2013dx']
        plt.arrow(y1[0], x1[0], -0.28, 0, color=sncolors['GRB'], head_width=0.1,
                  head_length=0.1)


    tempx = zip(temp1, temp10)
    tempy = zip(temp2, temp10)
    temp1 = [x[0] for x in tempx if x[1] == False]
    temp2 = [x[0] for x in tempy if x[1] == False]
    xavgs, yavgs = avgtrend3(temp2, temp1)
    plt.plot(xavgs, yavgs, color=sncolors['GRB'], linestyle='solid', linewidth=2)
    #if OHscale == 'PP04_O3N2':
    lcb, ucb, xavgs = avgtrend4(temp2, temp1)
    plt.fill_between(xavgs, lcb, ucb, color=sncolors['GRB'], alpha=0.2)


    #if i == 2:
    #       legend = plt.legend(loc='upper right', fontsize=fs-9, frameon=False, numpoints=1)

    #if OHscale in gzk1[0].keys():
    #       temp1 = [x["LgSFR_spec"][0] for x in gzk2]
    #       temp2 = [x[OHscale][0] for x in gzk1]
    #       temp3 = [x[OHscale][0]-x[OHscale][1] for x in gzk1]
    #       temp4 = [x[OHscale][2]-x[OHscale][0] for x in gzk1]
    #       temp5 = [x["LgSFR_spec"][0]-x["LgSFR_spec"][3] for x in gzk2]
    #       temp6 = [x["LgSFR_spec"][4]-x["LgSFR_spec"][0] for x in gzk2]
    #       plt.errorbar(temp1, temp2, xerr=[temp5, temp6], yerr=[temp3, temp4], fmt='g^', markeredgecolor='none')
    #       plt.annotate('12gzk', xy=(temp1[0]+0.1, temp2[0]-0.1), color='green')

    temp1 = [x["logsSFR"][0] for x in clean1 if x["SNtype"] == "uncertain"]
    temp2 = [x[OHscale][0] for x in clean1 if x["SNtype"] == "uncertain"]
    temp3 = [x[OHscale][0]-x[OHscale][1] for x in clean1
             if x["SNtype"] == "uncertain"]
    temp4 = [x[OHscale][2]-x[OHscale][0] for x in clean1
             if x["SNtype"] == "uncertain"]
    temp5 = [x["logsSFR"][0]-x["logsSFR"][3] for x in clean1
             if x["SNtype"] == "uncertain"]
    temp6 = [x["logsSFR"][4]-x["logsSFR"][0] for x in clean1
             if x["SNtype"] == "uncertain"]
    point7 = plt.errorbar(temp2, temp1, xerr=[temp3, temp4], yerr=[temp5, temp6],
                          fmt='g^', fillstyle='none', label='uncertain/weired')
    
    xerr_1 =np.sqrt((0.434*slsn_t4['e_SFR']/slsn_t4['SFR'])**2+slsn_t4['e_logM*']**2)
    xerr_2 =np.sqrt((0.434*slsn_t4['E_SFR']/slsn_t4['SFR'])**2+slsn_t4['E_logM*']**2)
    xerr_8 = [xerr_1, xerr_2]
    point8 = plt.errorbar(slsn_z_dat[OHscale], np.log10(slsn_t4['SFR'])-slsn_t4['logM*'], xerr=[slsn_z_dat['ne_'+OHscale], slsn_z_dat['pe_'+OHscale]], yerr=xerr_8, marker='*', color=sncolors['slsn'], label='SLSN Ic', linestyle='None')
    #if i == 0:
    #       first_legend = plt.legend(handles=[point1, point2, point3, point4],
    #loc='upper right', fontsize=fs-12, frameon=False, numpoints=1)
    #if i == 1:
    #       second_legend = plt.legend(handles=[point5, point6, point7],
    #loc='upper right', fontsize=fs-12, frameon=False, numpoints=1)
    #if i == 2:
    #       thrid_legend = plt.legend(handles=[point0, line1],
    #loc='upper right', fontsize=fs-12, frameon=False, numpoints=1)

# plt.show()
outfile = outpdf+'4.pdf'
plt.savefig(outfile, format='pdf')
plt.close(fig)


f = open(outpkl+'.pkl', 'wb')
pkl.dump(clean, f)
f.close()

'''
def main():
    #inpymcz = '/home/users/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_101916.pkl'
    #inpymcz = '/home/users/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_011717.pkl'
    #inpymcz = '/home/users/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_012617.pkl'
    #inpymcz = '/home/users/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_021717.pkl'
    #inpymcz = '/home/users/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_022717.pkl'
    #inpymcz = '/home/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_030317.pkl'
    #inpymcz = '/home/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_040317.pkl'
    #inpymcz = '/home/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_071717.pkl'
    #inpymcz = '/home/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_072117.pkl'
    #inpymcz = '/home/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_072617.pkl'
    inpymcz = 'platefitflux_072617.pkl'
    #infit = '/home/users/sh162/work/SED/python/SED_test_101916.pkl'
    #infit = '/home/users/sh162/work/SED/python/SED_test_012617.pkl'
    #infit = '/home/sh162/work/SED/python/SED_test_022717.pkl'
    #infit = '/home/sh162/work/SED/galex/SED_sdssgalex_022717.pkl'
    #infit = '/home/sh162/work/SED/fromDan/SED_all_041717.pkl'
    #infit = '/home/sh162/work/SED/fromDan/SED_all_071717.pkl'
    #infit = '/home/sh162/work/SED/fromDan/SED_all_072117.pkl'
    infit = 'SED_all_072117.pkl'
    #idfile = '/home/users/sh162/work/SED/PTF_SDSS_DR8.txt'
    #idfile = '/home/users/sh162/work/SED/PTF_SDSS_012617.txt'
    #idfile = '/home/sh162/work/SED/PTF_SDSS_022717.txt'
    #idfile = '/home/sh162/work/SED/fromDan/PTF_SDSS_041717.txt'
    #idfile = '/home/sh162/work/SED/fromDan/PTF_SDSS_071717.txt'
    #idfile = '/home/sh162/work/SED/fromDan/PTF_SDSS_072117.txt'
    idfile = 'PTF_SDSS_072117.txt'
    #inlvl = '/home/sh162/work/SED/fromDan/lvlsz.sav'
    inlvl = 'lvlsz.sav'
    #ingrb1 = '/home/users/sh162/work/platefit/platefit/4pyMCZ/python/GRB_110216.pkl'
    #ingrb1 = '/home/users/sh162/work/platefit/platefit/4pyMCZ/python/GRB_012017.pkl'
    #ingrb1 = '/home/users/sh162/work/platefit/platefit/4pyMCZ/python/GRB_021017.pkl'
    #ingrb1 = '/home/sh162/work/platefit/platefit/4pyMCZ/python/GRB_021717.pkl'
    #ingrb1 = '/home/sh162/work/platefit/platefit/4pyMCZ/python/GRB_031317.pkl'
    #ingrb1 = '/home/sh162/work/platefit/platefit/4pyMCZ/python/GRB_083017.pkl'
    ingrb1 = 'GRB_083017.pkl'
    #ingrb2 = '/home/users/sh162/work/platefit/platefit/4pyMCZ/GRB/GRBMsSFR.txt'
    #ingrb2 = '/home/users/sh162/work/platefit/platefit/4pyMCZ/GRB/GRBMsSFR_020817.txt'
    #ingrb2 = '/home/sh162/work/platefit/platefit/4pyMCZ/GRB/GRBMsSFR_030117.txt' # the new SN-GRB is not in this file!!! and thus is excluded from all plots
    #ingrb2 = '/home/sh162/work/platefit/platefit/4pyMCZ/GRB/GRBMsSFR_083017.txt' # the new SN-GRB is not included!!!
    ingrb2 = 'GRBMsSFR_083017.txt'
    #in12gzk1 = '/home/users/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_100716.pkl'
    #in12gzk1 = '/home/sh162/work/platefit/platefit/4pyMCZ/python/platefitflux_100716_new.pkl'
    #in12gzk2 = '/home/sh162/work/SED/12gzk/SED_nsa_12gzk.pkl'
    #insdss = '/home/sh162/work/SDSS/intervals.txt'
    insdss = 'intervals.txt'
    #insfsq = '/home/sh162/work/SED/fromDan/sfsq.txt'
    insfsq = 'sfsq.txt'
    #inohsq = '/home/sh162/work/SED/fromDan/ohsq.txt'
    inohsq = 'ohsq.txt'
    #insSFRsq = '/home/sh162/work/SED/fromDan/sSFRsq.txt'
    insSFRsq = 'sSFRsq.txt'
    outpdf = 'OH_MB_060418_all_'
    outpkl = 'OH_MB_121217_all'

    inslsn1 = 'apjaa3522t1_mrt.txt' #contains coordinates, redshift, magnitude, time of SN
    inslsn2 = 'apjaa3522t2_mrt.txt' #contains magnitude, flux density, observation date, instrument
    inslsn3 = 'apjaa3522t3_mrt.txt' #contains observation date, flag, setup, exposure sequence, sky position angle
    inslsn4 = 'apjaa3522t4_mrt.txt' #contains integrated galaxy magnitude, SFR, galaxy mass, extinction
    slsn_z = 'SLSN_z.txt'

    OH_MB(inpymcz, infit, idfile, inlvl, ingrb1, ingrb2, insdss, insfsq, inohsq,
          insSFRsq, outpdf, outpkl, inslsn1, inslsn2, inslsn3, inslsn4, slsn_z)


if __name__ == '__main__':
    main()
'''
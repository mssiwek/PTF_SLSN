# README
Our project comparing metallicities and SFRs of PTF SLSN, Ic, Ic-BL and SN-GRB hosts

This repo contains the code to generate all plots and tables comparing PTF SLSN/Ic/Ic-BL host galaxies. All the files in this repo are described in this README file. 

Most of the original data (e.g. SLSN fluxes and metallicities from Perley et al. 2016) is also contained in this directory, along with scripts to convert the original tables do formats suitable for processing with pyMCZ and other code.

# Data

* ## GRB line fluxes
* ## PTF SN Ic/Ic-BL line fluxes
* ## PTF SLSN line fluxes
* ## GRB metallicities
* ## PTF SN Ic/Ic-BL metallicities
* ## PTF SLSN metallicities

# Code to generate tables and plots

* ## BPT Diagram
** Code to generate plot:** 

BPT_071918.py

** Input: **

platefitfile = 'platefitflux_072117.txt'
SNinfofile = 'SNinfo_072617.txt'
GRBfile = 'GRBflux_083017_shan_astro9.txt'
SDSSfile = 'sdss_meas.txt'
SDSSerr = 'sdss_err.txt'
SLSNfile = 'SLSN_i_flux.txt'

** Output: **
Some SLSNe are flagged as they have upper limit measurements. If the code is run with those flagged hosts included, the output file is BPT_071918.pdf, if excluded  BPT_071918_noflags.pdf

* ## 
* ## 
* ## 




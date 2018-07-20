# README
Our project comparing metallicities and SFRs of PTF SLSN, Ic, Ic-BL and SN-GRB hosts

This repo contains the code to generate all plots and tables comparing PTF SLSN/Ic/Ic-BL host galaxies. Most of the original data (e.g. SLSN fluxes and metallicities from Perley et al. 2016, or P16) is also here, along with scripts to convert the original tables into formats suitable for processing with pyMCZ and other code.

All the SLSN host data is all from P16, so the paper is here too: `'Perley_2016_ApJ_830_13.pdf'`. 


# Data

* GRB line fluxes: `'GRBflux_083017_shan_astro9.txt'` -- there is another file with the same fluxes (`'GRBflux_083017.txt'`), but it has a slightly different format which produces errors when used as input in the below scripts.

* PTF SN Ic/Ic-BL line fluxes: 

`'platefitflux_072117.txt'` with header and observing info, such as instrument and region of host galaxy that was observed. 

`'ptfflux_072117.txt'` no header, just plain line fluxes and errors


* PTF SLSN line fluxes: original table 5 (P16) `'apjaa3522t5_mrt.txt'`, which has been processed to filter out SLSNe II and make the format (e.g. column headers) consistent with other line flux files. The script that does this is `'format_t5_t6'`, and produces the SLSN line flux files `'slsn_i_flux.txt'` (SLSNe Ic only) and `'slsn_flux.txt'` (all SLSNe published in P16).


* GRB metallicities: `'grboh_083017.txt'`, contains the following metallicity calibrations and their uncertainties: D13_N2S2_O3S2 (-/+), KD02comb (-/+), PP04_O3N2 (-/+), M08_N2Ha (-/+), E(B-V) (-/+), M13_O3N2 (-/+)


* PTF SN Ic/Ic-BL metallicities: `'ptfoh_072617.txt'`, however for generating the metallicity distributions (`'OH_dist_all.py'`) we want to have files containing SNe Ic and Ic-BL metallicity separately. The script `'separate_Ic_Icbl_z.py'` does this and saves the following files:

* `'ptf_unc_z.txt'`: contains host metallicities of SNe with unclear classification

* `'ptf_icbl_z.txt'`: host metallicities of SNe Ic-BL

* `'ptf_ic_z.txt'`: host metallicities of SNe Ic



* PTF SLSN metallicities: original table 6 (P16) `'apjaa3522t6_mrt.txt'`, which has been processed to filter out SLSNe II and make the format (e.g. column headers) consistent with other line flux files. The script that does this is `'format_t5_t6'`, and produces the SLSN line flux files `'slsn_i_z.txt'` (SLSNe Ic only) and `'slsn_z.txt'` (all SLSNe published in P16). The following calibrations are available: KD02comb, PP04_O3N2, M08_N2Ha, M13_O3N2 -- we want to eventually calculate those ourselves with pyMCZ, however we are waiting to hear back from Dan how he treated his upper limits before we can do this. 


# Code to generate tables and plots

* ## BPT Diagram
**Code to generate plot:** 

`BPT_071918.py`

**Input:**

* platefitfile = `'platefitflux_072117.txt'`
Emission line fluxes of the PTF SN hosts

* SNinfofile = `'Sninfo_072617.txt'`
SN subtypes

* GRBfile = `'GRBflux_083017_shan_astro9.txt'`
Emission line fluxes of the SN-GRB hosts as compiled from the literature

* SDSSfile = `'sdss_meas.txt'`
Line fluxes of 500 SDSS galaxies used as a comparison sample 

* SDSSerr = `'sdss_err.txt'`
Uncertainties of the above SDSS galaxy line fluxes

* SLSNfile = `'SLSN_i_flux.txt'`
Emission line flux + uncertainties of 18 type Ic SLSNe (P16)

**Output:**
Some SLSNe are flagged as they have upper limit measurements. If the code is run with those flagged hosts included (use the switch 'noflags' in the script), the output file is `BPT_071918.pdf`, if excluded  `BPT_071918_noflags.pdf`











* ## Redshift-Magnitude, Mass-Metallicity, Mass-SFR and 
sSFR-Metallicity plots

**Code to generate plots:** `’OH_MB_061118.py’`

**Input:**

* inpymcz = `'platefitflux_072617.pkl'`
A dictionary that contains emission line fluxes and metallicities as measured by pyMCZ, for the PTF Ic/Ic-BL hosts. This file is generated by OHdist_083017.py (not in this directory) from the ascii files that are output by pyMCZ. For more info on this file see documentation on the PTF host paper (Modjaz+ in prep): https://github.com/nyusngroup/PTFhostspaper

* infit = `'SED_all_072117.pkl'`
A dictionary that contains SED fitting results including the stellar masses and SFRs for the PTF hosts. See the documents in PTF paper documentation (Modjaz+ in prep: https://github.com/nyusngroup/PTFhostspaper) on SED fitting for how this file is created.

* idfile = `'PTF_SDSS_072117.txt'`
The photometry for SED fitting is downloaded from the SDSS and the hosts are labeled by the SDSS objIDs for that purpose. Use this file to cross match the SDSS objIDs to the PTF names. Four PTF hosts are outside of the SDSS footprints and their objids are hardwired to be 1-4. If the SDSS spectrum is not available, the specobjid column is 0 in this file.

* inlvl = `'sdss/lvlsz.sav'`
A file sent by Dan that contains the stellar masses and SFRs for the LVL hosts.

* ingrb1 = `'GRB_083017.pkl'`
A dictionary that contains emission line fluxes and metallicities as measured by pyMCZ, for the SN-GRB hosts. This file is generated by OHdist_083017.py from the ascii files that are output by pyMCZ (see above).

* ingrb2 = `'GRBMsSFR_083017.txt'`
An ascii file that records the stellar masses and SFRs of the SN-GRB hosts as compiled from the literature. See more notes inside this file for where do these numbers come from.

* insdss = `'sdss/intervals.txt'`
Average trends for the SDSS galaxies. This file is generated by running the jupyter notebook, M-Z relation from the SDSS galaxies.ipynb, which locates in ~/work/SDSS/. See the note file in that directory.

* insfsq = `'sdss/sfsq.txt'`
LVL and SDSS data 

* inohsq = `'sdss/ohsq.txt'`
SDSS and LVL metallicities: LVL only in one scale, SDSS in 4 different scales (PP04, M08, KD02, M13).

* insSFRsq = `'sdss/sSFRsq.txt'`
Average trends of the SF sequence, the M-Z relation, and the sSFR-metallicity relation, for the PTF host galaxies, respectively. These files are all generated by running the jupyter notebook, M-Z relation from the SDSS galaxies.ipynb, which locates in ~/work/SDSS/.

* inslsn1 = `'apjaa3522t1_mrt.txt'` 
Table 1 from P16, contains coordinates, redshift, magnitude, time of SN

* inslsn2 = `'apjaa3522t2_mrt.txt'` 
Table 2 from P16, contains magnitude, flux density, observation date, instrument

* inslsn3 = `'apjaa3522t3_mrt.txt'` 
Table 3 from P16, contains observation date, flag, setup, exposure sequence, sky position angle

* inslsn4 = `'apjaa3522t4_mrt.txt'` 
Table 4 from P16, contains integrated galaxy magnitude, SFR, galaxy mass, extinction

* inslsn5_orig = `'apjaa3522t5_mrt.txt'`
Table 5 from P16, contains integrated galaxy magnitude, SFR, galaxy mass, extinction

* slsn_i_z = `'slsn_i_z.txt'`
Metallicities of SLSNe from table 6 (P16), contains only those of type Ic

**Output:**

* `OH_MB_061118_all_0.pdf`: Redshift vs. Magnitude

* `OH_MB_061118_all_1.pdf`: Host galaxy mass vs. Metallicity in 4 scales (KD02comb, PP04_O3N2, M08_N2Ha, M13_O3N2)

* `OH_MB_061118_all_2.pdf`: Mass vs. SFR

* `OH_MB_061118_all_3.pdf`: sSFR vs. Metallicity in 4 scales (KD02comb, PP04_O3N2, M08_N2Ha, M13_O3N2)

* `OH_MB_061118_all_4.pdf`: Like figure `OH_MB_061118_all_3.pdf`, but with axes reversed (Metallicity vs. sSFR).






* ## Metallicity distributions

**Code to generate plot:** `'OH_dist_all.py'`

**Input:**

* slsn_z = `'slsn_i_z.txt'`
Metallicities of SLSNe Ic hosts. In this file the type II SLSNe that appear in the original table 6 in P16 have been filtered out.

* grb_z = `'grboh_083017.txt'`
Metallicities of GRB hosts. 

* ic_z = `'ptf_ic_z.txt'`
Metallicities of SNe Ic hosts. 

* icbl_z = `'ptf_icbl_z.txt'`
Metallicities of SNe Ic-BL hosts. 


**Output:**

`'OH_dist_062918_only_I.pdf'` - 4 panels showing histograms and cumulative metallicity distributions of SNe Ic, Ic-BL, GRB and SLSNe Ic hosts in 4 metallicity scales (KD02comb, PP04_O3N2, M08_N2Ha, M13_O3N2).




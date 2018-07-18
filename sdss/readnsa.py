from astropy.table import Table

def readnsa(nsafile, idfile):
	nsa = Table.read(nsafile, format='fits')
	plates = nsa['PLATE']
	fiberids = nsa['FIBERID']
	mjds = nsa['MJD']
	with open(idfile, 'w') as f:
		for i in range(len(nsa)):
			f.write(str(plates[i]).ljust(10)+str(fiberids[i]).ljust(10)+str(mjds[i]).ljust(10)+'\n')

def main():
	nsafile = '/data/sh162/NYU/NSA/nsa_v0_1_2.fits'
	idfile = '/home/sh162/work/SDSS/nsa_id.txt'
	readnsa(nsafile, idfile)

if __name__ == '__main__':
	main()

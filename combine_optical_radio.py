###combine data (VLASS/FIRST, VLASS/LS-DR9 etc)

import numpy as np
from astropy.table import Table, join, unique

#############################################################################
#############################################################################
###parameters

spfile = '../data/vlass-first_specidxs.fits'
lsfile = '../data/vlass1_x_lsdr9_lite.fits'

#############################################################################
#############################################################################
###functions

#############################################################################
#############################################################################
###main


###load data
spidx = Table.read(spfile)
lsdr9 = Table.read(lsfile)


###best lsdr9 match
lsdr9.sort('dist_arcsec')
best_ls = unique(lsdr9, 'Component_name')


###add in spec idx
xm_spec = join(best_ls, spidx, keys='Component_name', join_type='inner')
xm_spec.remove_columns(names=['RA', 'DEC'])

###write data
xm_spec.write('../data/best_vlass_x_lsdr9_with_spidx.fits')



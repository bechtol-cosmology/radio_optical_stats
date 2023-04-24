###combine VLASS and FIRST component catalogs
###account for single components in one catalog modelled by multiple in the other
###estimate spectral indices
###search around each FIRST component for VLASS components
###upper limit on search ~70''? restrict to matches within component geometry (maj/2, min/2 by pa?)

###max FIRST Maj == 180'', 76 with Maj > 60'' and 1,476 with maj > 30''
### len FIRST x VLASS (all matches):
##### r=90''; 876,250
##### r=30''; 707,050
##### r=15''; 635,487
### just do all matches and subset later
###for each match obtain separation and pos angle

import numpy as np
from astropy.table import Table, join, hstack, unique
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from collections import Counter

#############################################################################
#############################################################################
###parameters

vflux_scale = 1/0.87

vfile = '../data/VLASS1QL_components_lite.fits'
ffile = '/Users/yjangordon/Documents/science/survey_data/FIRST/catalogues/FIRST_catalog_2014_VizieR-VIII-92.fits'

fcols = ['FIRST', 'RA', 'DEC', 'Fint', 'Fpeak', 'Rms', 'Maj', 'Min', 'PA']
vcols = ['Component_name', 'RA', 'DEC', 'Total_flux', 'E_Total_flux',
         'DC_Maj', 'DC_Min', 'DC_PA']
vc2 = ['Component_name', 'RA', 'DEC', 'Peak_flux', 'Isl_rms',
       'Total_flux', 'E_Total_flux', 'DC_Maj', 'DC_Min', 'DC_PA']

#############################################################################
#############################################################################
###functions

def quad_sum(x):
    'return quadrature sum of array'
    qs = np.sqrt(np.sum(np.square(x)))
    
    return qs


def spec_idx(s1, s2, nu1, nu2, e_1=0, e_2=0, return_err=False):
    'estimate the spectral index of a source based on two flux measurements at different frequencies'
    ###error calculation based on eq 3 from A&A 630, A83 (2019)
    s1,s2 = np.array(s1), np.array(s2)
    
    srat = s1/s2
    nurat = nu1/nu2
    
    alpha = np.log(srat)/np.log(nurat)
    
    if return_err==True:
        e_1, e_2 = np.array(e_1), np.array(e_2)
        e_alpha = (1/(np.abs(nurat)))*np.sqrt((e_1/s1)**2 + (e_2/s2)**2)
        spidx = (alpha, e_alpha)
    else:
        spidx = alpha
        
    return spidx


def find_all_matches(d1, d2, searchrad=30*u.arcsec,
                     dname1='1', dname2='2',
                     acol1='RA', dcol1='DEC',
                     acol2='RA', dcol2='DEC',
                     posu1=('deg', 'deg'),
                     posu2=('deg', 'deg'),
                     sepunit='arcsec',
                     sepcol='match_dist',
                     pacol='match_pa',
                     ndp=2):
    'find all matches within a search radius'
    ###setup positional catalogs, ensure units in place
    if d1[acol1].unit is None:
        d1[acol1].unit = posu1[0]
    if d1[dcol1].unit is None:
        d1[dcol1].unit = posu1[1]
    if d2[acol2].unit is None:
        d2[acol2].unit = posu2[0]
    if d2[dcol2].unit is None:
        d2[dcol2].unit = posu2[1]
    
    cat1 = SkyCoord(ra=d1[acol1], dec=d1[dcol1])
    cat2 = SkyCoord(ra=d2[acol1], dec=d2[dcol2])
    
    ###add in NN_dist for each table
    nn1 = match_coordinates_sky(cat1, cat1, nthneighbor=2)[1].to(sepunit)
    nn2 = match_coordinates_sky(cat2, cat2, nthneighbor=2)[1].to(sepunit)
    d1['nn_dist'] = np.round(nn1, ndp)
    d2['nn_dist'] = np.round(nn2, ndp)
    
    xmatch = cat1.search_around_sky(cat2, seplimit=searchrad)
    
    idx1 = xmatch[1]
    idx2 = xmatch[0]
    sep = np.round(xmatch[2].to(sepunit), ndp)
    
    outdata = hstack(tables=[d1[idx1], d2[idx2]],
                     table_names=[dname1, dname2])
    
    outdata[sepcol] = sep
    
    ###add in position angle of match
    if acol1==acol2:
        acol1 = '_'.join([acol1, dname1])
        acol2 = '_'.join([acol2, dname2])
    if dcol1==dcol2:
        dcol1 = '_'.join([dcol1, dname1])
        dcol2 = '_'.join([dcol2, dname2])

    p1 = SkyCoord(ra=outdata[acol1], dec=outdata[dcol1])
    p2 = SkyCoord(ra=outdata[acol2], dec=outdata[dcol2])
    
    ###set posang to between 0 and 180
    posang = np.array(p1.position_angle(p2).to('deg'))
    posang[posang>=180] = posang[posang>=180]-180
    posang = np.round(posang, ndp)*u.deg
    
    outdata[pacol] = posang
    
    outdata.meta = {}
    
    return outdata


def sum_fluxes(data, group_by, sumcol, errcol,
               outsumcol='summed_flux',
               outerrcol='e_summed_flux',
               outgroupcol='n_matches',
               roundto=3,
               keytype=str):
    'sum fluxes of multiple matches'
    
    dset = data[[group_by, sumcol, errcol]].to_pandas()
    grouped = dset.groupby(group_by)
    summed = grouped[sumcol].agg('sum')
    e_summed = grouped[errcol].agg(quad_sum)
    sumdat = Table({group_by: list(summed.keys().astype(keytype)),
                    outsumcol: list(summed.values)})
    e_sumdat = Table({group_by: list(e_summed.keys().astype(keytype)),
                      outerrcol: list(np.round(e_summed.values, roundto))})
                    
    ###add unit
    sumdat[outsumcol].unit = data[group_by].unit
    matched = join(data, sumdat, keys=group_by, join_type='left')
    matched = join(matched, e_sumdat, keys=group_by, join_type='left')
    
    ###add in number of matches
    groupcount = Counter(np.array(matched[group_by]).astype(keytype))
    gcount = Table({group_by: list(groupcount.keys()),
                    outgroupcol: list(groupcount.values())})
    
    matched = join(matched, gcount, keys=group_by, join_type='left')
    
    ###subset unique and required columns
    matched = unique(matched, group_by)[[group_by, outsumcol, outerrcol, outgroupcol]]
    
    return matched

#############################################################################
#############################################################################
###main


vlass = Table.read(vfile)
first = Table.read(ffile)


fxv = find_all_matches(d1=first[fcols], d2=vlass[vcols],
                       searchrad=90*u.arcsec,
                       dname1='FIRST', dname2='VLASS')

fxv = fxv[(fxv['match_dist']<3) | ((np.abs(fxv['match_pa']-fxv['PA'])<30) & (fxv['match_dist']<fxv['Maj']/2))]
fxv.sort('FIRST')



matched = sum_fluxes(data=fxv, group_by='FIRST',
                     sumcol='Total_flux',
                     errcol='E_Total_flux',
                     outsumcol='VLASS_flux',
                     outerrcol='e_VLASS_flux',)

###get specindex info
matched = join(unique(fxv[['FIRST', 'Fint', 'Rms']], 'FIRST'), matched,
               keys='FIRST', join_type='inner')
matched['VLASS_flux'] = np.round(matched['VLASS_flux']*vflux_scale, 3)
spidx = spec_idx(s1=matched['Fint'], s2=matched['VLASS_flux'],
                 e_1=matched['Rms'], e_2=matched['e_VLASS_flux'],
                 nu1=1400, nu2=3000, return_err=True)
matched['spidx'] = np.round(spidx[0], 3)
matched['e_spidx'] = np.round(spidx[1], 3)


###matched is a list of FIRST components with spec idx from matching with VLASS
##invert: give list of VLASS components with spec idx from matching with first
vfirst = join(vlass[vc2], unique(fxv[['Component_name', 'FIRST']], 'Component_name'),
              keys='Component_name', join_type='inner')
vfirst = join(vfirst, matched[['FIRST', 'n_matches', 'spidx', 'e_spidx']]) ###spidx of VLASS components
fvlass = join(first, matched[['FIRST', 'n_matches', 'spidx', 'e_spidx']]) ###spidx of FIRST components
fvlass.meta = {}


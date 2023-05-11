###collate data on known lensed radio quasars
###some can be grabbed from NED, others may need looking at papers
###what to do when flux errors not provided? assume 5sigma?

import numpy as np
from astropy.table import Table, vstack, hstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.constants import c
from astroquery.ned import Ned

#####################################################################################
#####################################################################################
###parameters
ned_qcols = ['Object Name', 'angDist', 'Type', 'Photometry Points', 'References']

#####################################################################################
#####################################################################################
###functions
def get_ned_radio_photometry(name, nu_max=100*u.GHz,
                             nu_unit='MHz',
                             outcols=['Frequency', 'Flux Density',
                                      'NED Uncertainty', 'NED Units',
                                      'Refcode', 'Comments']):
    'obtain radio photometry for a named object in NED'
    photo = Ned.get_table(name, table='photometry')
    photo['Frequency'] = photo['Frequency'].to(nu_unit)
    photo = photo[photo['Frequency']<nu_max]
    photo = photo[outcols]
    
    return photo


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
    
    
def catalog_lens_image_spec_idxs(data, imname='NED_Object_Name',
                                 scol='Flux Density', escol='NED Uncertainty',
                                 nucol='Frequency', ndp=3):
    'make an output table of spectral index for lens images'
    ###make empty lists to append output data to
    names, spec_indices, s_flag, e_spec, e_flag, numin, numax = [], [], [], [], [], [], []
    imlist = np.unique(data[imname])
    for name in imlist:
        f_s, f_e = 0, 0
        photo = data[data[imname]==name]
        ###only positive fluxes and multiple frequency points
        photo = photo[photo[scol]>0]
        nu1, nu2 = np.min(photo[nucol]), np.max(photo[nucol])
        ##obtain data points to use
        d_1 = photo[photo[nucol]==nu1]
        d_2 = photo[photo[nucol]==nu2]
        if len(d_1)>1 or len(d_2)>1:
            f_s=1 ###updates flag if need to average multiple flux values
        s1 = np.nanmean(d_1[scol])
        s2 = np.nanmean(d_2[scol])
        e1 = [i.removeprefix('+/-') for i in d_1[escol]]
        e2 = [i.removeprefix('+/-') for i in d_2[escol]]
        for i in range(len(e1)):
            if len(e1[i])>0:
                e1[i] = float(e1[i])
            else:
                e1[i] = s1/5
                f_e = 1
        for i in range(len(e2)):
            if len(e2[i])>0:
                e2[i] = float(e2[i])
            else:
                e2[i] = s2/5
                f_e = 1
        e1 = np.nanmean(e1)
        e2 = np.nanmean(e2)
        spidx = spec_idx(nu1=nu1, nu2=nu2, s1=s1, s2=s2,
                         e_1=e1, e_2=e2, return_err=True)
        names.append(name)
        spec_indices.append(np.round(spidx[0], ndp))
        e_spec.append(np.round(spidx[1], ndp))
        s_flag.append(f_s)
        e_flag.append(f_e)
        numin.append(nu1)
        numax.append(nu2)
        
    ###make table
    outdata = Table({imname: names, 'spidx': spec_indices, 'e_spidx': e_spec,
                     'fluxes_averaged': s_flag, 'error_is_upper_limit': e_flag,
                     'freq_low': numin, 'freq_high': numax})
                     
    outdata = outdata[(outdata['freq_low']!=outdata['freq_high'])]
                     
    return outdata

#####################################################################################
#####################################################################################
###main
###set up list of NED query results
lensed_radio_list = Table.read('../../radio_lenses/data/known_lensed_radio_MMartinez.csv')


lensphot, missing = [], []
ned_res = {}
for row in lensed_radio_list:
    name = row['Name']
    pos = SkyCoord(ra=row['RA (ICRS)'], dec=row['Dec (ICRS)'], unit='deg')
    qres = Ned.query_region(pos, radius='10arcsec')
    respos = SkyCoord(ra=np.array(qres['RA']), dec=np.array(qres['DEC']), unit='deg')
    qres['angDist'] = np.round(pos.separation(respos).to('arcsec'), 3)
    qres.sort('angDist')
    qres = qres[(qres['Type']=='G_Lens') | (qres['Type']=='Q_Lens')] ##subsets only lens images
    ned_res[name] = qres
    if len(qres[qres['Photometry Points']>0])==0:
        missing.append(name)
    else:
        lensphot.append(name)


###get photometric data
photodat = []
for name in lensphot:
    ndata = ned_res[name]
    ndata = ndata[ndata['Photometry Points']>0]
    objlist = list(ndata['Object Name'])
    seps = list(ndata['angDist'])
    for i in range(len(objlist)):
        oname = objlist[i]
        osep = seps[i]
        pdat = get_ned_radio_photometry(oname)
        pdat.meta = {} ###cleans up metadata for vstacking
        if len(pdat)==0:
            missing.append(name)
        else:
            oinfo = Table({'Martinez_Name': len(pdat)*[name],
                           'NED_Object_Name': len(pdat)*[oname],
                           'angDist': len(pdat)*[osep*u.arcsec]})
            pdat = hstack([oinfo, pdat])
            photodat.append(pdat)
    
photodat = vstack(photodat)


specdat = catalog_lens_image_spec_idxs(data=photodat)

#photodat.write('../../radio_lenses/data/NED_radio_photometry_lensed_qso_images.xml', format='votable')
#specdat.write('../../radio_lenses/data/specidx_from_NED_photometry.xml', format='votable')

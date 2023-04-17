###stack xmatched tables into single table and add meta data

import numpy as np
from astropy.table import Table, vstack
from astropy import units as u


###############################################################
###############################################################
###paramters

dpath = '../data'
file_list = ['vlass1_x_lsdr9_1.fits',
             'vlass1_x_lsdr9_2.fits',
             'vlass1_x_lsdr9_3.fits',
             'vlass1_x_lsdr9_4.fits']

coldict = {'Component_name': {'description': 'VLASS component name', 'unit': None},
           'ls_id': {'description': 'Unique LS object ID', 'unit': None},
           'dec': {'description': 'Declination at equinox J2000', 'unit': 'deg'},
           'ra': {'description': 'Right ascension at equinox J2000', 'unit': 'deg'},
           'elat': {'description': 'Ecliptic Latitude', 'unit': 'deg'},
           'elon': {'description': 'Ecliptic Longitude', 'unit': 'deg'},
           'glat': {'description': 'Galactic Latitude', 'unit': 'deg'},
           'glon': {'description': 'Galactic Longitude', 'unit': 'deg'},
           'mjd_max': {'description': 'Maximum Modified Julian Date of observations used to construct the model of this object', 'unit': None},
           'mjd_min': {'description': 'Minimum Modified Julian Date of observations used to construct the model of this object', 'unit': None},
           'ref_id': {'description': 'Reference catalog identifier for this star; Tyc1*1,000,000+Tyc2*10+Tyc3 for Tycho2; sourceid for Gaia DR2 and SGA', 'unit': None},
           'brickid': {'description': 'Brick ID [1,662174]', 'unit': None},
           'blob_nea_g': {'description': 'Blob-masked noise equivalent area in g', 'unit': None},
           'blob_nea_r': {'description': 'Blob-masked noise equivalent area in r', 'unit': None},
           'blob_nea_z': {'description': 'Blob-masked noise equivalent area in z', 'unit': None},
           'bx': {'description': 'X position (0-indexed) of coordinates in the brick image stack (i.e. in the e.g. legacysurvey--image-g.fits.fz coadd file)', 'unit': None},
           'by': {'description': 'Y position (0-indexed) of coordinates in the brick image stack (i.e. in the e.g. legacysurvey--image-g.fits.fz coadd file)', 'unit': None},
           'dchisq_1': {'description': 'Difference in chi^2 between successively more-complex model fits: PSF, REX, DEV, EXP, SER. The difference is versus no source', 'unit': None},
           'dchisq_2': {'description': 'Difference in chi^2 between successively more-complex model fits: PSF, REX, DEV, EXP, SER. The difference is versus no source', 'unit': None},
           'dchisq_3': {'description': 'Difference in chi^2 between successively more-complex model fits: PSF, REX, DEV, EXP, SER. The difference is versus no source', 'unit': None},
           'dchisq_4': {'description': 'Difference in chi^2 between successively more-complex model fits: PSF, REX, DEV, EXP, SER. The difference is versus no source', 'unit': None},
           'dchisq_5': {'description': 'Difference in chi^2 between successively more-complex model fits: PSF, REX, DEV, EXP, SER. The difference is versus no source', 'unit': None},
           'dec_ivar': {'description': 'Inverse variance of DEC, excluding astrometric calibration errors', 'unit': None},
           'dered_flux_g': {'description': 'Dereddened g-band flux', 'unit': None},
           'dered_flux_r': {'description': 'Dereddened r-band flux', 'unit': None},
           'dered_flux_w1': {'description': 'Dereddened w1-band flux', 'unit': None},
           'dered_flux_w2': {'description': 'Dereddened w2-band flux', 'unit': None},
           'dered_flux_w3': {'description': 'Dereddened w3-band flux', 'unit': None},
           'dered_flux_w4': {'description': 'Dereddened w4-band flux', 'unit': None},
           'dered_flux_z': {'description': 'Dereddened z-band flux', 'unit': None},
           'dered_mag_g': {'description': 'Dereddened g-band magnitude', 'unit': 'mag'},
           'dered_mag_r': {'description': 'Dereddened r-band magnitude', 'unit': 'mag'},
           'dered_mag_w1': {'description': 'Dereddened w1-band magnitude', 'unit': 'mag'},
           'dered_mag_w2': {'description': 'Dereddened w2-band magnitude', 'unit': 'mag'},
           'dered_mag_w3': {'description': 'Dereddened w3-band magnitude', 'unit': 'mag'},
           'dered_mag_w4': {'description': 'Dereddened w4-band magnitude', 'unit': 'mag'},
           'dered_mag_z': {'description': 'Dereddened z-band magnitude', 'unit': 'mag'},
           'ebv': {'description': 'Galactic extinction E(B-V) reddening from SFD98, used to compute the mw_transmission_ columns', 'unit': None},
           'fiberflux_g': {'description': 'Predicted g-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing', 'unit': None},
           'fiberflux_r': {'description': 'Predicted r-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing', 'unit': None},
           'fiberflux_z': {'description': 'Predicted z-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing', 'unit': None},
           'fibertotflux_g': {'description': 'Predicted g-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing', 'unit': None},
           'fibertotflux_r': {'description': 'Predicted r-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing', 'unit': None},
           'fibertotflux_z': {'description': 'Predicted z-band flux within a fiber of diameter 1.5 arcsec from this object in 1 arcsec Gaussian seeing', 'unit': None},
           'flux_g': {'description': 'Model flux in g', 'unit': None},
           'flux_ivar_g': {'description': 'Inverse variance of flux_g', 'unit': None},
           'flux_ivar_r': {'description': 'Inverse variance of flux_r', 'unit': None},
           'flux_ivar_w1': {'description': 'Inverse variance of flux_w1 (AB)', 'unit': None},
           'flux_ivar_w2': {'description': 'Inverse variance of flux_w2 (AB)', 'unit': None},
           'flux_ivar_w3': {'description': 'Inverse variance of flux_w3 (AB)', 'unit': None},
           'flux_ivar_w4': {'description': 'Inverse variance of flux_w4 (AB)', 'unit': None},
           'flux_ivar_z': {'description': 'Inverse variance of flux_z', 'unit': None},
           'flux_r': {'description': 'Model flux in r', 'unit': None},
           'flux_w1': {'description': 'WISE model flux in W1 (AB system)', 'unit': None},
           'flux_w2': {'description': 'WISE model flux in W2 (AB system)', 'unit': None},
           'flux_w3': {'description': 'WISE model flux in W3 (AB system)', 'unit': None},
           'flux_w4': {'description': 'WISE model flux in W4 (AB system)', 'unit': None},
           'flux_z': {'description': 'Model flux in z', 'unit': None},
           'fracflux_g': {'description': 'Profile-weighted fraction of the flux from other sources divided by the total flux in g (typically [0,1])', 'unit': None},
           'fracflux_r': {'description': 'Profile-weighted fraction of the flux from other sources divided by the total flux in r (typically [0,1])', 'unit': None},
           'fracflux_w1': {'description': 'Profile-weighted fraction of the flux from other sources divided by the total flux in w1 (typically [0,1])', 'unit': None},
           'fracflux_w2': {'description': 'Profile-weighted fraction of the flux from other sources divided by the total flux in w2 (typically [0,1])', 'unit': None},
           'fracflux_w3': {'description': 'Profile-weighted fraction of the flux from other sources divided by the total flux in w3 (typically [0,1])', 'unit': None},
           'fracflux_w4': {'description': 'Profile-weighted fraction of the flux from other sources divided by the total flux in w4 (typically [0,1])', 'unit': None},
           'fracflux_z': {'description': 'Profile-weighted fraction of the flux from other sources divided by the total flux in z (typically [0,1])', 'unit': None},
           'fracin_g': {'description': 'Fraction of a sources flux within the blob in g, near unity for real sources', 'unit': None},
           'fracin_r': {'description': 'Fraction of a sources flux within the blob in r, near unity for real sources', 'unit': None},
           'fracin_z': {'description': 'Fraction of a sources flux within the blob in z, near unity for real sources', 'unit': None},
           'fracmasked_g': {'description': 'Profile-weighted fraction of pixels masked from all observations of this object in g, strictly between [0,1]', 'unit': None},
           'fracmasked_r': {'description': 'Profile-weighted fraction of pixels masked from all observations of this object in r, strictly between [0,1]', 'unit': None},
           'fracmasked_z': {'description': 'Profile-weighted fraction of pixels masked from all observations of this object in z, strictly between [0,1]', 'unit': None},
           'gaia_a_g_val': {'description': 'Gaia line-of-sight extinction in the G band', 'unit': None},
           'gaia_astrometric_excess_noise': {'description': 'Gaia astrometric excess noise', 'unit': None},
           'gaia_astrometric_excess_noise_sig': {'description': 'Gaia astrometric excess noise uncertainty', 'unit': None},
           'gaia_astrometric_sigma5d_max': {'description': 'Gaia longest semi-major axis of the 5-d error ellipsoid', 'unit': None},
           'gaia_astrometric_weight_al': {'description': 'Gaia astrometric weight along scan direction', 'unit': None},
           'gaia_e_bp_min_rp_val': {'description': 'Gaia line-of-sight reddening E(BP-RP)', 'unit': None},
           'gaia_phot_bp_mean_flux_over_error': {'description': 'Gaia BP signal-to-noise', 'unit': None},
           'gaia_phot_bp_mean_mag': {'description': 'Gaia BP mag', 'unit': None},
           'gaia_phot_bp_rp_excess_factor': {'description': 'Gaia BP/RP excess factor', 'unit': None},
           'gaia_phot_g_mean_flux_over_error': {'description': 'Gaia G band signal-to-noise', 'unit': None},
           'gaia_phot_g_mean_mag': {'description': 'Gaia G band mag', 'unit': None},
           'gaia_phot_rp_mean_flux_over_error': {'description': 'Gaia RP band signal-to-noise', 'unit': None},
           'gaia_phot_rp_mean_mag': {'description': 'Gaia RP mag', 'unit': None},
           'galdepth_g': {'description': 'As for psfdepth_g but for a galaxy (0.45 arcsec exp, round) detection sensitivity', 'unit': None},
           'galdepth_r': {'description': 'As for psfdepth_r but for a galaxy (0.45 arcsec exp, round) detection sensitivity', 'unit': None},
           'galdepth_z': {'description': 'As for psfdepth_z but for a galaxy (0.45 arcsec exp, round) detection sensitivity', 'unit': None},
           'g_r': {'description': 'Computed (g-r) color', 'unit': 'mag'},
           'htm9': {'description': 'HTM index (order 9 => ~10 arcmin size)', 'unit': None},
           'mag_g': {'description': 'Converted g magnitude', 'unit': 'mag'},
           'mag_r': {'description': 'Converted r magnitude', 'unit': 'mag'},
           'mag_w1': {'description': 'Converted w1 magnitude', 'unit': 'mag'},
           'mag_w2': {'description': 'Converted w2 magnitude', 'unit': 'mag'},
           'mag_w3': {'description': 'Converted w3 magnitude', 'unit': 'mag'},
           'mag_w4': {'description': 'Converted w4 magnitude', 'unit': 'mag'},
           'mag_z': {'description': 'Converted z magnitude', 'unit': 'mag'},
           'mw_transmission_g': {'description': 'Galactic transmission in g filter in linear units [0, 1]', 'unit': None},
           'mw_transmission_r': {'description': 'Galactic transmission in r filter in linear units [0, 1]', 'unit': None},
           'mw_transmission_w1': {'description': 'Galactic transmission in w1 filter in linear units [0, 1]', 'unit': None},
           'mw_transmission_w2': {'description': 'Galactic transmission in w2 filter in linear units [0, 1]', 'unit': None},
           'mw_transmission_w3': {'description': 'Galactic transmission in w3 filter in linear units [0, 1]', 'unit': None},
           'mw_transmission_w4': {'description': 'Galactic transmission in w4 filter in linear units [0, 1]', 'unit': None},
           'mw_transmission_z': {'description': 'Galactic transmission in wz filter in linear units [0, 1]', 'unit': None},
           'nea_g': {'description': 'Noise equivalent area in g', 'unit': None},
           'nea_r': {'description': 'Noise equivalent area in r', 'unit': None},
           'nea_z': {'description': 'Noise equivalent area in z', 'unit': None},
           'nest4096': {'description': 'HEALPIX index (Nsides 4096, Nest scheme => ~52 arcsec size', 'unit': None},
           'objid': {'description': 'Catalog object number within this brick; a unique identifier hash is release,brickid,objid; objid spans [0,N-1] and is contiguously enumerated within each brick', 'unit': None},
           'parallax_ivar': {'description': 'Reference catalog inverse-variance on parallax', 'unit': None},
           'parallax': {'description': 'Reference catalog parallax', 'unit': None},
           'pmdec_ivar': {'description': 'Reference catalog inverse-variance on pmdec', 'unit': None},
           'pmdec': {'description': 'Reference catalog proper motion in the Dec direction', 'unit': None},
           'pmra_ivar': {'description': 'Reference catalog inverse-variance on pmra', 'unit': None},
           'pmra': {'description': 'Reference catalog proper motion in the RA direction', 'unit': None},
           'psfdepth_g': {'description': 'For a 5 sigma point source detection limit in g, 5/(sqrt psfdepth_g) gives flux in nanomaggies and (-2.5[log10(5/(sqrt psfdepth_g))-9]) gives corresponding AB magnitude', 'unit': None},
           'psfdepth_r': {'description': 'For a 5 sigma point source detection limit in r, 5/(sqrt psfdepth_r) gives flux in nanomaggies and (-2.5[log10(5/(sqrt psfdepth_r))-9]) gives corresponding AB magnitude', 'unit': None},
           'psfdepth_w1': {'description': 'As for psfdepth_g (and also on the AB system) but for WISE W1', 'unit': None},
           'psfdepth_w2': {'description': 'As for psfdepth_g (and also on the AB system) but for WISE W2', 'unit': None},
           'psfdepth_w3': {'description': 'As for psfdepth_g (and also on the AB system) but for WISE W3', 'unit': None},
           'psfdepth_w4': {'description': 'As for psfdepth_g (and also on the AB system) but for WISE W4', 'unit': None},
           'psfdepth_z': {'description': 'For a 5 sigma point source detection limit in z, 5/(sqrt psfdepth_z) gives flux in nanomaggies and (-2.5[log10(5/(sqrt psfdepth_z))-9]) gives corresponding AB magnitude', 'unit': None},
           'psfsize_g': {'description': 'Weighted average PSF FWHM in the g band', 'unit': 'arcsec'},
           'psfsize_r': {'description': 'Weighted average PSF FWHM in the r band', 'unit': 'arcsec'},
           'psfsize_z': {'description': 'Weighted average PSF FWHM in the z band', 'unit': 'arcsec'},
           'ra_ivar': {'description': 'Inverse variance of RA (no cosine term!), excluding astrometric calibration errors', 'unit': None},
           'random_id': {'description': 'Random ID in the range 0.0 => 100.0', 'unit': None},
           'rchisq_g': {'description': 'Profile-weighted chi^2 of model fit normalized by the number of pixels in g', 'unit': None},
           'rchisq_r': {'description': 'Profile-weighted chi^2 of model fit normalized by the number of pixels in r', 'unit': None},
           'rchisq_w1': {'description': 'Profile-weighted chi^2 of model fit normalized by the number of pixels in w1', 'unit': None},
           'rchisq_w2': {'description': 'Profile-weighted chi^2 of model fit normalized by the number of pixels in w2', 'unit': None},
           'rchisq_w3': {'description': 'Profile-weighted chi^2 of model fit normalized by the number of pixels in w3', 'unit': None},
           'rchisq_w4': {'description': 'Profile-weighted chi^2 of model fit normalized by the number of pixels in w4', 'unit': None},
           'rchisq_z': {'description': 'Profile-weighted chi^2 of model fit normalized by the number of pixels in z', 'unit': None},
           'ref_epoch': {'description': 'Reference catalog reference epoch (eg, 2015.5 for Gaia DR2)', 'unit': None},
           'ring256': {'description': 'HEALPIX index (Nsides 256, Ring scheme => ~14 arcmin size)', 'unit': None},
           'r_z': {'description': 'Computed (r-z) color', 'unit': 'mag'},
           'sersic_ivar': {'description': 'Inverse variance of sersic', 'unit': None},
           'sersic': {'description': 'Power-law index for the Sersic profile model (type=SER)', 'unit': None},
           'shape_e1_ivar': {'description': 'Inverse variance of shape_e1', 'unit': None},
           'shape_e1': {'description': 'Ellipticity component 1 of galaxy model for galaxy type type', 'unit': None},
           'shape_e2_ivar': {'description': 'Inverse variance of shape_e2', 'unit': None},
           'shape_e2': {'description': 'Ellipticity component 2 of galaxy model for galaxy type type', 'unit': None},
           'shape_r_ivar': {'description': 'Inverse variance of shape_r', 'unit': None},
           'shape_r': {'description': 'Half-light radius of galaxy model for galaxy type type (>0)', 'unit': None},
           'snr_g': {'description': 'Signal-to-Noise ratio in g', 'unit': None},
           'snr_r': {'description': 'Signal-to-Noise ratio in r', 'unit': None},
           'snr_w1': {'description': 'Signal-to-Noise ratio in w1', 'unit': None},
           'snr_w2': {'description': 'Signal-to-Noise ratio in w2', 'unit': None},
           'snr_w3': {'description': 'Signal-to-Noise ratio in w3', 'unit': None},
           'snr_w4': {'description': 'Signal-to-Noise ratio in w4', 'unit': None},
           'snr_z': {'description': 'Signal-to-Noise ratio in z', 'unit': None},
           'w1_w2': {'description': 'Computed (w1-w2) color', 'unit': 'mag'},
           'w2_w3': {'description': 'Computed (w2-w3) color', 'unit': 'mag'},
           'w3_w4': {'description': 'Computed (w3-w4) color', 'unit': 'mag'},
           'wise_x': {'description': 'X position of coordinates in the brick image stack that corresponds to wise_coadd_id (see the DR9 updates page for transformations between wise_x and bx)', 'unit': None},
           'wise_y': {'description': 'Y position of coordinates in the brick image stack that corresponds to wise_coadd_id (see the DR9 updates page for transformations between wise_y and by)', 'unit': None},
           'z_w1': {'description': 'Computed (z-w1) color', 'unit': 'mag'},
           'allmask_g': {'description': 'Bitwise mask set if the central pixel from all images satisfy each condition in g as cataloged on the DR9 bitmasks page', 'unit': None},
           'allmask_r': {'description': 'Bitwise mask set if the central pixel from all images satisfy each condition in r as cataloged on the DR9 bitmasks page', 'unit': None},
           'allmask_z': {'description': 'Bitwise mask set if the central pixel from all images satisfy each condition in z as cataloged on the DR9 bitmasks page', 'unit': None},
           'anymask_g': {'description': 'Bitwise mask set if the central pixel from any image satisfies each condition in g as cataloged on the DR9 bitmasks page', 'unit': None},
           'anymask_r': {'description': 'Bitwise mask set if the central pixel from any image satisfies each condition in r as cataloged on the DR9 bitmasks page', 'unit': None},
           'anymask_z': {'description': 'Bitwise mask set if the central pixel from any image satisfies each condition in z as cataloged on the DR9 bitmasks page', 'unit': None},
           'brick_primary': {'description': 'True if the object is within the brick boundary', 'unit': None},
           'fitbits': {'description': 'Bitwise mask detailing pecularities of how an object was fit, as cataloged on the DR9 bitmasks page', 'unit': None},
           'gaia_astrometric_n_good_obs_al': {'description': 'Gaia number of good astrometric observations along scan direction', 'unit': None},
           'gaia_astrometric_n_obs_al': {'description': 'Gaia number of astrometric observations along scan direction', 'unit': None},
           'gaia_astrometric_params_solved': {'description': 'Which astrometric parameters were estimated for a Gaia source', 'unit': None},
           'gaia_duplicated_source': {'description': 'Gaia duplicated source flag', 'unit': None},
           'gaia_phot_bp_n_obs': {'description': 'Gaia BP number of observations', 'unit': None},
           'gaia_phot_g_n_obs': {'description': 'Gaia g band number of observations', 'unit': None},
           'gaia_phot_rp_n_obs': {'description': 'Gaia RP band number of observations', 'unit': None},
           'gaia_phot_variable_flag': {'description': 'Gaia photometric variable flag', 'unit': None},
           'maskbits': {'description': 'Bitwise mask indicating that an object touches a pixel in the coadd/*/*/*maskbits* maps, as cataloged on the DR9 bitmasks page', 'unit': None},
           'nobs_g': {'description': 'Number of images that contribute to the central pixel in g: filter for this object (not profile-weighted)', 'unit': None},
           'nobs_r': {'description': 'Number of images that contribute to the central pixel in r: filter for this object (not profile-weighted)', 'unit': None},
           'nobs_w1': {'description': 'Number of images that contribute to the central pixel in w1: filter for this object (not profile-weighted)', 'unit': None},
           'nobs_w2': {'description': 'Number of images that contribute to the central pixel in w2: filter for this object (not profile-weighted)', 'unit': None},
           'nobs_w3': {'description': 'Number of images that contribute to the central pixel in w3: filter for this object (not profile-weighted)', 'unit': None},
           'nobs_w4': {'description': 'Number of images that contribute to the central pixel in w4: filter for this object (not profile-weighted)', 'unit': None},
           'nobs_z': {'description': 'Number of images that contribute to the central pixel in z: filter for this object (not profile-weighted)', 'unit': None},
           'release': {'description': 'Integer denoting the camera and filter set used, which will be unique for a given processing run of the data, as documented at https://www.legacysurvey.org/dr9/catalogs/', 'unit': None},
           'wisemask_w1': {'description': 'W1 bitmask as cataloged on the DR9 bitmasks page', 'unit': None},
           'wisemask_w2': {'description': 'W2 bitmask as cataloged on the DR9 bitmasks page', 'unit': None},
           'brickname': {'description': 'Name of brick, encoding the brick sky position, eg 1126p222 near RA=112.6, Dec=+22.2', 'unit': None},
           'ref_cat': {'description': 'Reference catalog source for this star: T2 for Tycho-2, G2 for Gaia DR2, L3 for the SGA, empty otherwise', 'unit': None},
           'type': {'description': 'Morphological model: PSF=stellar, REX=round exponential galaxy, DEV=deVauc, EXP=exponential, SER=Sersic, DUP=Gaia source fit by different model.', 'unit': None},
           'wise_coadd_id': {'description': 'unWISE coadd brick name (corresponding to the, e.g., legacysurvey--image-W1.fits.fz coadd file) for the center of each object', 'unit': None},
           'dist_arcsec': {'description': 'angular separation of match', 'unit': 'arcsec'}}

litecols = ['Component_name', 'ls_id', 'ra', 'dec', 'type',
            'dered_mag_g', 'dered_mag_r', 'dered_mag_z',
            'dered_mag_w1', 'dered_mag_w2', 'dered_mag_w3',
            'dered_mag_w4', 'snr_g', 'snr_r', 'snr_z',
            'snr_w1', 'snr_w2', 'snr_w3', 'snr_w4', 'g_r',
            'r_z', 'z_w1', 'w1_w2', 'w2_w3', 'w3_w4',
            'ebv', 'dist_arcsec']


###############################################################
###############################################################
###functions

def concat_multiple_tables(filelist, filedir='.',
                           rename_cols=None,
                           new_names=None):
    'vstack tables and renames columns if required'
    data_list = []
    for f in filelist:
        fname = '/'.join([filedir, f])
        print(f'loading data from {fname}')
        dat = Table.read(fname)
        data_list.append(dat)
    
    data = vstack(data_list)
    
    if rename_cols is not None and new_names is not None:
        print('renaming columns')
        for i in range(len(rename_cols)):
            oname = rename_cols[i]
            nname = new_names[i]
            print(f'  {oname} -> {nname}')
        data.rename_columns(names=rename_cols, new_names=new_names)
        
    return data


def add_column_meta(data, colmeta,
                    desckey='description',
                    unitkey='unit',
                    verbose=True):
    'add in column descriptions and units to data'
    metacols = list(colmeta.keys())
    if verbose==True:
        print('adding metadata to table columns:')
    for col in data.colnames:
        if col in metacols:
            cdesc = colmeta[col][desckey]
            cunit = colmeta[col][unitkey]
            data[col].description = cdesc
            data[col].unit = cunit
            if verbose==True:
                print(f' {col}:')
                print(f'   unit={cunit}')
                print(f'   description={cdesc}')
        else:
            if verbose==True:
                print(f' no metadata for {col}')
    
    return


###############################################################
###############################################################
###main

data = concat_multiple_tables(filelist=file_list, filedir=dpath,
                              rename_cols=['t1_component_name'],
                              new_names=['Component_name'])
                            
add_column_meta(data=data, colmeta=coldict)

dlite = data[litecols]

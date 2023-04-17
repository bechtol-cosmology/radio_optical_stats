###verify host reliability

import numpy as np, matplotlib.pyplot as plt, sys
from astropy.table import Table, join, unique
from astropy.coordinates import SkyCoord
from astropy import units as u
from collections import Counter
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
sys.path.append('/Users/yjangordon/Documents/science/science_code/astro_tools/')
from astro_tools import *

plt.interactive(True)

####plot sep vs spec idx

###############################################################
###############################################################
###parameters

savefigs = True
outdir = '../data/quickplots'
dpi = 250

#ddir = ('../../VLASS_DRAGNs/code/test_code/cirada_wrapper/'
#        + 'output_files_2000deg2_test/')
#sfile = ddir + 'sources.fits'
#dfile = ddir + 'dragns.fits'
#hfile = ddir + 'supplementary_data/hosts.fits'
#lrfile = ddir + 'supplementary_data/all_LR_matches.fits'
#randfile = ddir + 'supplementary_data/dragns_no_core_random_pos_awise_seps.fits'
#wsfile = ('/Users/yjangordon/Documents/science/survey_data/'
#          + 'WISE/AllWISE/egal_1000sqdeg_sample.fits')
#d_awise_xmatch_file = ('../../VLASS_DRAGNs/data/Oct2022_rerun_data/'
#                       + 'supplementary_data/'
#                       + 'dragns_no_core_awise_real_rand_crossmatches.fits')
#rwfile = ddir + 'supplementary_data/random_40k_allwise.fits'

vfile = '../data/VLASS1QL_components_lite.fits'
realfile = '../data/VLASS-QL1xDES-DR2.fits'
randfile = '../data/random_positions_x_DES-DR2.fits'
#realfile = '../data/test_data/vlass_x_des_litesample.fits'
#randfile = '../data/test_data/random_x_des_litesample.fits'


sepbins = np.linspace(0, 30, 30)

n_awise = 747634026
a_sky = 4*np.pi*u.steradian
#awise_density = n_awise/a_sky.to('deg2')
 
###############################################################
###############################################################
###functions

def find_closest(data, sepcol='angDist', namecol='Component_name'):
    'find the closest match from all within a cross match table'
    best = data.copy()
    best.sort(sepcol)
    best = unique(best, namecol, keep='first')
    
    return best


def random_sample_of_xmatches(data, namecol='Component_name',
                              n_rand=1000):
    'select cross matches for random sample of inputs'
    sample = unique(data[[namecol]], namecol)
    sample = sample[np.random.choice(len(sample), n_rand, replace=False)]
    sample = join(sample, data, keys=namecol, join_type='left')
    
    return sample
    

def footrint_from_xmatch_data(data, namecol='Component_name',
                              search_rad=None, sepcol='angDist'):
    'determine sky footprint of catalog of random xmatches'
    ###if search_rad is provided use this, else determine from data
    if search_rad is None:
        if sepcol is None or sepcol not in data.columns:
            print('Neither search_rad or sepcol provided. One of these is required to determine sky footprint')
            return
        if data[sepcol].unit is not None:
            sepu = data[sepcol].unit
        else:
            print(f"column '{sepcol}' contains no unit information, assuming units are arcsec")
            sepu = u.arcsec
        search_rad = data[sepcol].max()*sepu
    
    ##number of search coords
    n_search = len(unique(data, namecol))
    footprint_area = n_search * np.pi * search_rad**2
    
    return footprint_area.to('deg2')


def midbins(bins):
    'returns middle value from bin edges for plotting'
    x = []
    for i in range(len(bins)-1):
        x0, x1 = bins[i], bins[i+1]
        midx = x0 + ((x1-x0)/2)
        x.append(midx)
    
    return np.array(x)


def expected_random(Npos, sepbins, source_density):
    'determine number of expected random contaminants at search radius r for Npos searches given a source density'
    ##eq 11 from Galvin+ (2020, MNRAS 497, 2730)
    radius = midbins(sepbins)
    binwidths = [sepbins[i+1]-sepbins[i] for i in range(len(sepbins)-1)]
    binwidths = np.array(binwidths)
    erand = Npos * source_density * 2 * np.pi * radius * binwidths
    
    return erand
    
    
def sky_footprint(amin, amax, dmin, dmax):
    'estimate sky area for square footprint'
    dra = amax-amin
    ddec = dmax-dmin
    
    f_ra = dra/360 ###fraction of RA range covered
    
    ddec = np.deg2rad(ddec)
    h = np.tan(ddec)
    
    area = 2*np.pi*h*f_ra*u.steradian
    
    return area.to('deg2')


def rel_from_sep(seps, source_density, sepbins=np.linspace(0, 30, 30),
                 sigma_smooth=1):
    'estimate p real from sep distributions alone'
    n = len(seps)
    nreal = np.histogram(seps, sepbins)[0]
    nrand = expected_random(Npos=n, sepbins=sepbins, source_density=source_density)
    
    ###emperical p
    p_r = nreal/(nrand+nreal)
    
    ###smooth and interp
    smoothed = gaussian_filter1d(p_r, sigma=sigma_smooth)
    
    p_interp = interp1d(midbins(sepbins), smoothed)
    
    return p_interp


def source_density_within_mag(mag_data, mag, footprint, dmag=0.25,
                              amin=120, amax=220, dmin=-5, dmax=5):
    'determine the density of sources with mag±dmag'
#    if footprint is None:
##        footprint = sky_footprint(amin=amin, amax=amax, dmin=dmin, dmax=dmax)
#        footrint_from_xmatch_data(mag_data, namecol='Component_name',
#                              search_rad=None, sepcol='angDist')
    
    mags = mag_data[(mag_data>mag-dmag) & (mag_data<mag+dmag)]
    n_sources = len(mags)
    sdens = n_sources/footprint
    
    return sdens
    
    
    

def rel_sep_by_mag(mag, mag_data, seps, footprint,
                   dmag=0.25,
                   sepbins=np.linspace(0, 30, 30),
                   sigma_smooth=2):
    'determine function p_r based on magnitude'
    ##determine source density
    source_density = source_density_within_mag(mag_data=mag_data, mag=mag, dmag=dmag,
                                               footprint=footprint)
    
    ###initial p_r(mag)
    pr_initial = rel_from_sep(seps=seps, sepbins=sepbins,
                              source_density=source_density.to('arcsec-2').value)
    
    ###smooth p_r
    x = midbins(sepbins)
    y_smoothed = gaussian_filter1d(pr_initial(x), sigma=sigma_smooth)
    
    ###interp to make pr(sep, mag)
    
    p_r_mag = interp1d(x, y_smoothed)
    
    return p_r_mag


def rel_mag_sep_for_data(data, mag_data, sep_data, magcol='W1mag', sepcol='Sep_AllWISE',
                         sepbins=np.linspace(0, 30, 30), dmag=0.5):
    'determine Rel (mag, sep) for each DRAGN in table'
    
    hseps = np.array(data[sepcol])
    hmags = np.array(data[magcol])
    x = midbins(sepbins)
    
    rel_r_mag = []
    for i in range(len(hmags)):
        w1mag = hmags[i]
        if w1mag>16:
            sig_smooth=1.5
        elif w1mag>15:
            sig_smooth=2
        else:
            sig_smooth=3
        wsep = hseps[i]
        if wsep<np.min(x):
            wsep=np.min(x)
        rel_m_r_i = rel_sep_by_mag(mag=w1mag, mag_data=mag_data,
                                   seps=sep_data, dmag=dmag,
                                   sigma_smooth=sig_smooth)(wsep)
        rel_r_mag.append(rel_m_r_i)
        
    rel_r_mag = np.array(rel_r_mag)
    
    return rel_r_mag


def quasi_rel(p_real, q0=0.6):
    'estimate Rel assuming only potential candidate (ignore other candidates); Eq 12 from McAlpine et al. 2012.'
    
    lr_proxy = p_real/(1-p_real)
    rel = lr_proxy/(lr_proxy+(1-q0))
    
    return rel


def compare_sep_dists(host_seps, random_seps, nrand, source_density,
                      sepbins=np.linspace(0, 30, 30),
                      figsize=(8,6),
                      rncol='C3', rnalpha=0.2, rnlab='from random position',
                      rlcol='C0', rlls='-', rllw=1.8, rllab='from VLASS',
                      mcol='k', mls=':', mlw=2, mlab='expected from source density',
                      xlab='angular separation [arcsec]',
                      ylab=r'$N$', fontsize=16, legloc=9):
    'plot sep dists for host matches, modelled random, and nearest'
    ###determine modelled random
    exprand = expected_random(Npos=nrand, sepbins=sepbins,
                              source_density=source_density.to('arcsec-2').value)
    
    x = midbins(sepbins)
    
    ###make figure
    plt.figure(figsize=figsize)
    plt.hist(random_seps, sepbins, color=rncol, alpha=rnalpha, label=rnlab)
    plt.hist(host_seps, sepbins, histtype='step', color=rlcol,
             ls=rlls, lw=rllw, label=rllab)
    plt.plot(x, exprand, c=mcol, ls=mls, lw=mlw, label=mlab)
    
    plt.legend(loc=legloc, fontsize=fontsize-3)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    
    plt.xlim(np.min(sepbins), np.max(sepbins))
    
    plt.grid(ls=':')
    
    return


def plot_mag_dists(real, random, magbins=np.linspace(10.5, 32.5, 50),
                   figsize=(8,6),
                   rncol='C3', rnalpha=0.2, rnlab='random matches',
                   rlcol='C0', rlls='-', rllw=1.8, rllab='closest VLASS',
                   xlab='W1 [mag]',
                   ylab=r'$n$', fontsize=16, legloc=2,
                   logy=True):
    'plot real and random magnitude distributions'
    
    plt.figure(figsize=figsize)
    plt.hist(random, magbins, color=rncol, alpha=rnalpha, label=rnlab,
             density=True)
    plt.hist(real, magbins, histtype='step', color=rlcol,
             ls=rlls, lw=rllw, label=rllab, density=True)
    
    plt.legend(loc=legloc, fontsize=fontsize-4)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    
    plt.xlim(np.min(magbins), np.max(magbins))
    
    if logy==True:
        plt.yscale('log')
    
    plt.grid(ls=':')
    
    return


def plot_exprand_by_mag(mag_data, sepbins=np.linspace(0, 30, 30),
                        dmag=0.25, sdens_all=130000*u.Unit('deg-2'),
                        footprint=0.21816616*u.Unit('deg2'),
                        mags=np.array([12, 13, 14, 15, 16, 17]),
                        amin=120, amax=220, dmin=-5, dmax=5,
                        cmap=plt.cm.viridis_r, figsize=(8,6),
                        c_all='k', lw_all=2, ls_all=':', lab_all='all sources',
                        ls_mag='-', legloc=4, band='W1',
                        xlab=r'angular separation [arcsec]',
                        ylab=r'Expected matches from a random position',
                        fontsize=14, logy=True):
    'make a plot showing the impact of mag limits on random matches'
    
    ###create colormap to sample line colors from
    lincolors = cmap(np.linspace(0, 1, len(mags)))
    x = midbins(sepbins)

    ###global expected random
    medmag = np.nanmedian(mag_data)
    magrange = np.nanmax(mag_data)-np.nanmin(mag_data)
#    if sdens_all is None: -- this needs fixing to be automated
#        sdens_all = source_density_within_mag(mag_data=mag_data, mag=medmag,
#                                              dmag=magrange, amin=amin,
#                                              amax=amax, dmin=dmin, dmax=dmax)
    erand = expected_random(Npos=1, sepbins=sepbins,
                            source_density=sdens_all.to('arcsec-2').value)
    print(sdens_all)
    ###setup figure
    plt.figure(figsize=figsize)
    plt.plot(x, erand, c=c_all, ls=ls_all, lw=lw_all, label=lab_all)
    
    
    
    ###iterate through mags
    for i in range(len(mags)):
        mag = mags[i]
        mag_c = lincolors[i]
        lab = f'{band} = {mag} ± {dmag}'
        sdens = source_density_within_mag(mag_data=mag_data, mag=mag, dmag=dmag,
                                          footprint=footprint)
        er_mag = expected_random(Npos=1, sepbins=sepbins,
                                 source_density=sdens.to('arcsec-2').value)
        plt.plot(x, er_mag, ls=ls_mag, lw=lw_all-0.5, color=mag_c, label=lab)
        
    plt.legend(loc=legloc, fontsize=fontsize-3)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.xlim(np.min(sepbins), np.max(sepbins))
    
    if logy == True:
        plt.yscale('log')
        
    plt.grid(ls=':')

    return


def plot_rel_by_sep_mags(seps, mag_data, footprint,
                         mags=[12, 13, 14, 15, 16, 17],
                         sepbins=np.linspace(0, 30, 30),
                         amin=120, amax=220, dmin=-5, dmax=5,
                         cmap=plt.cm.viridis_r,
                         dmag=0.5, pr_all=None, lab_all='all mags',
                         ls_all=':', lw_all=2, c_all='k',
                         ls_mag='-', figsize=(8,6),
                         fontsize=14, legloc=1, band='W1',
                         xlab=r'angular separation, $r$, [arcsec]',
                         ylab=r'$P_{\rm{real}} (\rm{mag}, r)$',
                         ):
    'make plot showing smoothed p(r, mag) functions for different mags'
    
    ###pre plot setup
    x = midbins(sepbins)
    lincolors = cmap(np.linspace(0, 1, len(mags)))
    
    ###make plot
    plt.figure(figsize=figsize)
    if pr_all is not None:
        x = x[(x>=np.min(pr_all.x)) & (x<=np.max(pr_all.x))]
        plt.plot(x, pr_all(x), ls=ls_all, lw=lw_all, c=c_all, label=lab_all)
    for i in range(len(mags)):
        mag = mags[i]
        if mag > 16:
            sig_smooth=1.5
        elif mag > 15:
            sig_smooth=2
        else:
            sig_smooth=3
        mag_c = lincolors[i]
        lab = f'{band} = {mag} ± {dmag}'
        p_rmag = rel_sep_by_mag(mag=mag, mag_data=mag_data, seps=seps, dmag=dmag,
                                sepbins=sepbins, sigma_smooth=sig_smooth,
                                footprint=footprint)
        y = p_rmag(x)
        plt.plot(x, y, ls=ls_mag, lw=lw_all-0.5, color=mag_c, label=lab)
    
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.xlim(np.min(sepbins), np.max(sepbins))
    plt.ylim(0, 1.05)

    plt.grid(ls=':')
    plt.legend(loc=legloc, fontsize=fontsize-3)
        
    return


###############################################################
###############################################################
###main


#vlass = Table.read(vfile)
real = Table.read(realfile)
random = Table.read(randfile)
best_real = find_closest(data=real)
best_random = find_closest(data=random)

footprint_real = footrint_from_xmatch_data(data=real)
footprint_rand = footrint_from_xmatch_data(data=random)

sdens_overall = len(random)/footprint_rand


###plotting params
sep_bins = np.linspace(0, 30, 60)
magbins = np.linspace(10.5, 32.5, 50)
eg_mags = [12, 13, 14, 15, 16, 17]
delta_mag = 0.4

#####plots:
####1) sep dist of all candidate matches with random and modelled random based on source density overlaid

compare_sep_dists(host_seps=np.array(real['angDist']),
                  random_seps=np.array(random['angDist']),
                  nrand=len(unique(random, 'Component_name')),
                  source_density=sdens_overall,
                  sepbins=sep_bins)
if savefigs == True:
    outname = '/'.join([outdir, 'seps_VLASSxDES.png'])
    plt.savefig(outname, dpi=dpi)
    plt.close()

###2) show random mag distribution and double host mag distribution
filters = ['g', 'r', 'i', 'z', 'Y']
filters = ['i']
for m in filters:
    mcol = ''.join([m, 'mag'])
    mlab = r'$m_{' + m + r'}$ [mag]'
    plot_mag_dists(real=np.array(best_real[mcol]),
                   random=np.array(random[mcol]),
                   xlab=mlab, magbins=magbins)
    if savefigs == True:
        fname = f'mags_{m}-band_VLASSxDES.png'
        outname = '/'.join([outdir, fname])
        plt.savefig(outname, dpi=dpi)
        plt.close()
        
    ###3) show magnitude distribution and impact on modelled random sep distribution that reduced source density given mag±dmag(e.g. 0.5) has
    eg_mags = np.array([20, 21, 22, 23, 24])
    delta_mag=0.5
    mag_data = random[mcol]
    band_label = r'$m_{' + m + r'}$'
    plot_exprand_by_mag(mag_data=mag_data, sepbins=sep_bins,
                        dmag=delta_mag, mags=eg_mags, band=band_label,
                        footprint=footprint_rand)
    if savefigs == True:
        fname = f'expected_random_matches_sep+{m}-band_VLASSxDES.png'
        outname = '/'.join([outdir, fname])
        plt.savefig(outname, dpi=dpi)
        plt.close()

    ###4) determine p_real given random AND magnitude
    p_r = rel_from_sep(seps=np.array(best_real['angDist']),
                       source_density=sdens_overall.to('arcsec-2').value,
                       sigma_smooth=0.6)
    plot_rel_by_sep_mags(seps=np.array(best_real['angDist']),
                         mag_data=mag_data, pr_all=p_r,
                         sepbins=sep_bins, mags=eg_mags,
                         dmag=delta_mag, footprint=footprint_rand,
                         band=band_label)
    if savefigs == True:
        fname = f'p-real_sep+{m}-band_VLASSxDES.png'
        outname = '/'.join([outdir, fname])
        plt.savefig(outname, dpi=dpi)
        plt.close()



###3) show magnitude distribution and impact on modelled random sep distribution that reduced source density given mag±dmag(e.g. 0.5) has.

#plot_exprand_by_mag(mag_data=mag_data, sepbins=sep_bins,
#                    dmag=delta_mag, mags=eg_mags)


####testing
#wsamp = Table.read(wsfile)
#sources = Table.read(sfile)
#dragns_all = Table.read(dfile)
#wise = Table.read(hfile)
#lrres = Table.read(lrfile)
#randseps = Table.read(randfile)
#rwise = Table.read(rwfile)
#
#p_awise = len(wsamp)/sky_footprint(amin=np.min(wsamp['RAJ2000']),
#                                   amax=np.max(wsamp['RAJ2000']),
#                                   dmin=np.min(wsamp['DEJ2000']),
#                                   dmax=np.max(wsamp['DEJ2000']))






###old testing
#
#
#dragns = dragns_all[dragns_all['Core_prom'].mask] ###keeps only dragns without core
#dhosts = dragns[~dragns['AllWISE'].mask]
#
####join randseps so only comparing with DRAGNS with hosts)
#randseps = join(dhosts[['Name']], randseps, keys='Name', join_type='inner')
#
#
####find all/best host candidate(s) based on sep alone
#dwise = join(dragns[['Name', 'LAS']], wise, join_type='left', keys='Name')
#dwise = dwise[~dwise['AllWISE'].mask]
#dwise.sort('Sep_AllWISE')
#dbest = unique(dwise, 'Name', keep='first')
#dwise.sort('Name')
#
#
####parameters for analysis
#
#eg_mags = [12, 13, 14, 15, 16, 17]
#mag_data = np.array(wsamp['W1mag'])
#sep_data = np.array(dhosts['Sep_AllWISE'])
#sep_bins = np.linspace(0, 30, 30)
#delta_mag = 0.4 ###0.4 mag bin width used by LR for determining distributions
#
#
#####plots:
####1) sep dist of all candidate hosts with random and modelled random based on source density overlaid
#
#compare_sep_dists(host_seps=sep_data,
#                  random_seps=np.array(randseps['angDist']),
#                  nrand=len(unique(randseps, 'Name')),
#                  source_density=p_awise,
#                  sepbins=sep_bins)
#
#
####2) show random mag distribution and double host mag distribution
#
#plot_mag_dists(real=np.array(dhosts['W1mag']),
#               random=np.array(randseps['W1mag']))
#
#
####3) show magnitude distribution and impact on modelled random sep distribution that reduced source density given mag±dmag(e.g. 0.5) has.
#
#plot_exprand_by_mag(mag_data=mag_data, sepbins=sep_bins,
#                    dmag=delta_mag, mags=eg_mags)
#
#
####4) determine p_real given random AND magnitude
#
#p_r = rel_from_sep(seps=sep_data,
#                   source_density=p_awise.to('arcsec-2').value, sigma_smooth=1.2)
####p_r(sep) returns estimate of probability that match is real given sep and source density!
#
#plot_rel_by_sep_mags(seps=sep_data, mag_data=mag_data, pr_all=p_r,
#                     sepbins=sep_bins, mags=eg_mags, dmag=delta_mag)
#
#
#
####5) compare p_real to Rel for LR IDs (DRAGNs without hosts only)
#
#
#####e.g.
##pr_12 = rel_sep_by_mag(mag=12, mag_data=np.array(wsamp['W1mag']),
##                       seps=np.array(dhosts['Sep_AllWISE']),
##                       dmag=0.25, sepbins=np.linspace(0, 30, 30),
##                       sigma_smooth=3)
####and rel for any dragn, d, is (dmag may need to be high (e.g. ~1) to replicate LR
#####W1>16 sm=1.5; elif W1> 15 sm=2; else sm=3
##rel_i = rel_sep_by_mag(mag=d['W1mag'], mag_data=np.array(wsamp['W1mag']),
##                       seps=np.array(dhosts['Sep_AllWISE']), dmag=0.25,
##                       sigma_smooth=sm)(d['Sep_AllWISE'])
#
#
####this takes a while, load a table of data if don't need to redo
#
#rcfile = ('/Users/yjangordon/Documents/science/Papers/active/VLASS_DRAGNs/'
#          + 'code/test_code/cirada_wrapper/output_files_2000deg2_test/'
#          + 'supplementary_data/host_rel_comp_dragns_no_core_11_11_2022.fits')
#relcomp = Table.read(rcfile)
#
##p_real = rel_mag_sep_for_data(data=dhosts, mag_data=mag_data, sep_data=sep_data,
##                              sepbins=sep_bins, dmag=delta_mag)
##qrel = quasi_rel(p_real, q0=0.6)
##wcand = join(wise[['Name', 'Sep_AllWISE', 'AllWISE']],
##             dhosts[['Name','LAS']], keys='Name', join_type='right')
##wcand = wcand[(wcand['Sep_AllWISE']<0.3*wcand['LAS'])]
##wcount = Counter(wcand['Name'])
##wcount = Table({'Name': list(wcount.keys()), 'n_host_cand': list(wcount.values())})
##
##dhosts['p_real_yg'] = np.round(p_real, 6)
##dhosts['qrel_yg'] = np.round(qrel, 6)
##
##relcomp = dhosts[['Name', 'Flux', 'E_Flux', 'LAS', 'E_LAS', 'AllWISE',
##                  'Sep_AllWISE', 'W1mag', 'e_W1mag', 'Rel', 'p_real_yg',
##                  'qrel_yg']]
##relcomp = join(relcomp, wcount, keys='Name', join_type='inner')
#
#reltol=0.1
#disagree = relcomp[((np.abs(relcomp['Rel']-relcomp['qrel_yg'])>reltol)
#                    & (relcomp['n_host_cand']==1))]
#
#
#
#####fraction without host ID -- move above plotting when done
#
#dmissing = dbest[(dbest['Sep_AllWISE']<=0.3*dbest['LAS'])]
#dmissing = join(dmissing, dragns[dragns['AllWISE'].mask][['Name']],
#                keys='Name', join_type='inner')
#dmissing = join(dmissing[['Name', 'AllWISE', 'W1mag',
#                          'e_W1mag', 'Sep_AllWISE']],
#                lrres, keys='AllWISE', join_type='left')


####need following numbers/data
##number of dragns with/without host ID
##number of dragns with/without AllWISE within 30''
##
##
##
## 884 DRAGNs (no core)
## 877 have at least 1 AllWISE counterpart within 30''
## 753 have at least 1 AllWISE counterpart within 0.3*LAS (and 30'')
## 738 have host ID'd by LR
##
## 146 (17%) have no host ID after LR
## 15 (1.7%) of these have a viable candidate host that was rejected by LR
## 7 of these not considered by the LR (at edge of image?)
## 1 removed because Rel > 1 (marginally, probably a rounding error)
## rest have v. low Rel.
## main take home here is 17% of DRAGNs have no host






####FWHM of p(r) for dragns = 6''
#### sigma = 1/2.355 * 6'' = 2.55 ''

###need to estimate reliability given magnitude:
#### i.e. for dragns use random source density given by mag ± 0.5 mag (or som bin size)
##estimate density from test sky area



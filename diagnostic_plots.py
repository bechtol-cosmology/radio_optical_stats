###make diagnostic plots

import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table, join
import seaborn as sns, corner

plt.interactive(1)
#############################################################################
#############################################################################
###parameters

sample_pair_plots = True
nsamp = 5000


#############################################################################
#############################################################################
###functions

def sep_v_spidx(sep, spidx, figsize=(8, 6),
                fontsize=15, plot_flat=True,
                plot_sepline=True, sepline=0.5,
                minsep=0.05, xmin=0.045, xmax=10,
                xtix=[0.1, 1, 10],
                ytix=[-3, -2, -1, 0, 1, 2, 3],
                ymin=-3, ymax=3, nbins=100,
                overlays=None,
                xlabel='radio-optical offset [arcsec]',
                ylabel='spectral index',
                cmap='viridis',
                lcol='k', sepls=':', sils='--', lw=1.8,
                seplab =  'radio-optical offset = ',
                flatlab = 'spectral index = ±0.5',
                grid=True):
    'make figuere showing radio/optical separation vs spectal index'
    ###ensure arrays and account for zero/incredibly small offsets
    sep = np.array(sep)
    sep[sep<minsep] = minsep
    spidx = np.array(spidx)
    
    ###setup bins for histogram
    xbins = np.logspace(np.log10(minsep), np.log10(xmax), nbins)
    ybins = np.linspace(ymin, ymax, nbins)
    
    ###make figure
    fig = plt.figure(figsize=figsize)
    
    plt.hist2d(sep, spidx, [xbins, ybins], norm=LogNorm(), cmap=cmap)
    
    if overlays is not None:
        if type(overlays)==list:
            for i in range(len(overlays)):
                plt.scatter(overlays['x'], overlays['y'], s=overlays['ms'],
                            color=overlays['color'], marker=overlays['marker'],
                            label=overlays['label'])
        else:
            plt.scatter(overlays['x'], overlays['y'], s=overlays['ms'],
                        color=overlays['color'], marker=overlays['marker'],
                        label=overlays['label'])
    
    if plot_sepline==True:
        seplab = seplab + str(sepline) + "''"
        plt.plot([sepline, sepline], [ymin, ymax], c=lcol, ls=sepls, lw=lw,
                 label=seplab)
    if plot_flat==True:
        plt.plot([xmin, xmax], [-0.5, -0.5], c=lcol, ls=sils, lw=lw,
                 label=flatlab)
        plt.plot([xmin, xmax], [0.5, 0.5], c=lcol, ls=sils, lw=lw)
    
    plt.legend()
    plt.xscale('log')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(xtix, xtix)
    plt.yticks(ytix, ytix)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    
    if grid==True:
        plt.grid(ls=':')
            
    return


def corner_plot(data, cols=None, **kwargs):
    if cols is None:
        cols = data.colnames
    
    plotdata = data[cols]
    for col in cols: ##filter only finite and real
        plotdata = plotdata[(~np.isnan(plotdata[col])) & (np.isfinite(plotdata[col]))]
    
    pdat = {}
    for col in cols:
        pdat[col] = np.array(plotdata[col])
        
    corner.corner(pdat, **kwargs)
    
    return


#############################################################################
#############################################################################
###main


data = Table.read('../data/best_vlass_x_lsdr9_with_spidx.fits')
photo_zs = Table.read('../data/best_vlass_x_lsdr9_photo_zs.fits')
data = data[data['n_matches']==1] ###eliminates bad spidx measurments
data = join(data, photo_zs, keys='ls_id', join_type='left')
qsos = Table.read('../../../../survey_data/lensed_quasars/lensedquasars_yg-update.fits')
lensed_radio = Table.read('../../radio_lenses/data/known_lensed_radio_MMartinez.csv')
ned_photo = Table.read('../../radio_lenses/data/NED_radio_photometry_lensed_qso_images.xml')
ned_spidx = Table.read('../../radio_lenses/data/specidx_from_NED_photometry.xml')
ned_spidx = join(ned_spidx, ned_photo[['NED_Object_Name', 'angDist']],
                 keys='NED_Object_Name', join_type='inner')




###test corner_plot
tcols = ['dist_arcsec', 'spidx', 'dered_mag_w1', 'zphot']
collabs = ['sep [arcsec]', r'$\alpha$', 'W1 [mag]', 'z']
corner_plot(data=pdat, cols=tcols, bins=50, color='C7', smooth=0.8,
            labels=collabs,
            label_kwargs={'fontsize': 14},
            axes_scale=['linear', 'linear', 'linear', 'linear'],
            levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


#####need to do plain cross matches with Lensed QSOs and Michaels list too
#ned_overlay = {'x': ned_spidx['angDist'], 'y': ned_spidx['spidx'],
#               'ms': 15, 'color': 'r', 'marker': 's', 'label': 'known (NED)'}
#
#sep_v_spidx(sep=data['dist_arcsec'], spidx=data['spidx'], overlays=ned_overlay)
#plt.legend(loc=9, fontsize=12, ncols=2, bbox_to_anchor=(0.5, 1.15), scatterpoints=3)

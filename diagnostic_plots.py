###make diagnostic plots

import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table

plt.interactive(1)
#############################################################################
#############################################################################
###parameters


#############################################################################
#############################################################################
###functions

def sep_v_spidx(sep, spidx, figsize=(7, 6),
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


#############################################################################
#############################################################################
###main


data = Table.read('../data/best_vlass_x_lsdr9_with_spidx.fits')
data = data[data['n_matches']==1] ###eliminates bad spidx measurments
qsos = Table.read('../../../../survey_data/lensed_quasars/lensedquasars_yg-update.fits')
lensed_radio = Table.read('../../radio_lenses/data/known_lensed_radio_MMartinez.csv')
ned_photo = Table.read('../../radio_lenses/data/NED_radio_photometry_lensed_qso_images.xml')
ned_spidx = Table.read('../../radio_lenses/data/specidx_from_NED_photometry.xml')




sep_v_spidx(sep=data['dist_arcsec'], spidx=data['spidx'])

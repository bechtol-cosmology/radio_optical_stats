### Query DES via CDS VizieR service from input table

import numpy as np, argparse
from astropy.table import Table, join, hstack, vstack
from astropy import units as u
from astroquery.xmatch import XMatch


#############################################################
#############################################################
###parameters

###dictionary of column definitions/units to augment data (make sure this is correct for the data being queried or set col_dict=None)

coldict = {'Component_name': {'unit': None,
                              'description': "Name of VLASS component"},
           'RAJ2000': {'unit': 'deg',
                      'description': 'Right ascension for the object, J2000 in ICRS system'},
           'DEJ2000': {'unit': 'deg',
                      'description': 'Declination for the object, J2000 in ICRS system'},
           'DES': {'unit': None,
                   'description': 'Identifier based on IAU format (DES JHHMMSS.ss+DDMMSS.s)'},
           'CoadID': {'unit': None,
                      'description': 'Unique identifier for the co-added objects'},
           'Aimg': {'unit': 'pix',
                    'description': 'Major axis size based on an isophotal model'},
           'Bimg': {'unit': 'pix',
                    'description': 'Minor axis size based on an isophotal model'},
           'PA': {'unit': 'deg',
                  'description': 'Position angle of source in J2000 coordinates, from non-windowed measurement'},
           'ExtClsCoad': {'unit': None,
                         'description': 'Classification from Sextractor quantities'},
           'ExtClsWavg': {'unit': None,
                         'description': 'Classification from Weighted average PSF magnitudes'},
           'gmag': {'unit': 'mag',
                    'description': 'g-band (AB) magnitude estimation'},
           'rmag': {'unit': 'mag',
                    'description': 'r-band (AB) magnitude estimation'},
           'imag': {'unit': 'mag',
                    'description': 'i-band (AB) magnitude estimation'},
           'zmag': {'unit': 'mag',
                    'description': 'z-band (AB) magnitude estimation'},
           'Ymag': {'unit': 'mag',
                    'description': 'Y-band (AB) magnitude estimation'},
           'e_gmag': {'unit': 'mag',
                      'description': 'gmag uncertainty'},
           'e_rmag': {'unit': 'mag',
                      'description': 'rmag uncertainty'},
           'e_imag': {'unit': 'mag',
                      'description': 'imag uncertainty'},
           'e_zmag': {'unit': 'mag',
                      'description': 'zmag uncertainty'},
           'e_Ymag': {'unit': 'mag',
                      'description': 'Ymag uncertainty'},
           'gFlag': {'unit': None,
                     'description': 'Cautionary flag for g-band (<4=well behaved objects)'},
           'rFlag': {'unit': None,
                     'description': 'Cautionary flag for r-band (<4=well behaved objects)'},
           'iFlag': {'unit': None,
                     'description': 'Cautionary flag for i-band (<4=well behaved objects)'},
           'zFlag': {'unit': None,
                     'description': 'Cautionary flag for z-band (<4=well behaved objects)'},
           'yFlag': {'unit': None,
                     'description': 'Cautionary flag for Y-band (<4=well behaved objects)'}}


#############################################################
#############################################################
###functions

def load_targets(filename, namecol='Component_name',
                 acol='RA', dcol='DEC',
                 posunits=('deg', 'deg')):
    'load targets to query'
    ###load data
    data = Table.read(filename)
    data = data[[namecol, acol, dcol]]
    
    ###ensure units in positions
    if data[acol].unit is None:
        data[acol].unit = posunits[0]
    if data[dcol].unit is None:
        data[dcol].unit = posunits[1]
    
    return data
    

def cds_xmatch(data, racol='RA', decol='DEC',
               maxsep=30*u.arcsec,
               cat2='vizier:II/371/des_dr2',
               timeout=1200,
               remove_poscols=True,
               sep_prec=3,
               column_dict=None,
               chunk_size=20000,
               namecol='Component_name'):
    'Query CDS X-match service'
    ###try to replace with async query (might not need to)
    xm = XMatch()
    xm.TIMEOUT = timeout
    
    ###split into chunks if large
    if len(data)>chunk_size:
        n_chunks = int(np.ceil(len(data)/chunk_size))
        print(f'upload data is large, breaking into chunks of <= {chunk_size} rows')
        xm_chunks = []
        for i in range(n_chunks):
            print('')
            print(f'Querying chunk {i+1}/{n_chunks}')
            dchunk = data[i*chunk_size: (i+1)*chunk_size]
            xmc = xm.query(cat1=dchunk, cat2=cat2,
                           max_distance=maxsep,
                           colRA1=racol, colDec1=decol)
            if len(xmc)>0:
                xm_chunks.append(xmc)
        xmatch = vstack(xm_chunks)
    else:
        xmatch = xm.query(cat1=data, cat2=cat2,
                          max_distance=maxsep,
                          colRA1=racol, colDec1=decol)
    
    ###add in column units/descriptions
    if column_dict is not None:
        collist = list(column_dict.keys())
        for col in xmatch.colnames:
            if col in collist:
                xmatch[col].unit = column_dict[col]['unit']
                xmatch[col].description = column_dict[col]['description']
                
    ###remove original poscols
    if remove_poscols==True:
        xmatch.remove_columns(names=[racol, decol])
    
    ###make angdist last column
    if xmatch.colnames[0]=='angDist':
        xcols = xmatch.colnames[1:] + ['angDist']
        xmatch = xmatch[xcols]
    
    ###add in sep col units and description
    xmatch['angDist'] = np.round(xmatch['angDist'], sep_prec)
    xmatch['angDist'].unit = maxsep.unit
    xmatch['angDist'].description = "angular separation between matched objects"
    
    xmatch.sort(namecol)
    
    return xmatch


def parse_args():
    "parse input args, i.e. target and config file names"
    parser = argparse.ArgumentParser(description="query CDS XMatch servis")
    parser.add_argument("targets",
                        help="table of target positions to query")
    parser.add_argument("--namecol", action='store', type=str,
                        default='Component_name',
                        help="target ID column in input data")
    parser.add_argument("--racol", action='store', type=str,
                        default='RA',
                        help="name of RA col in target data")
    parser.add_argument("--decol", action='store', type=str,
                        default='DEC',
                        help="name of Decl. col in target data")
    parser.add_argument("--outdir", action='store', type=str, default='.',
                        help="directory to write output file to")
    parser.add_argument("--outfile", action='store', type=str,
                        default='CDS-Xmatch_results.fits',
                        help="output filename")
    parser.add_argument("--search_radius", action='store', type=str,
                        default='30arcsec',
                        help="search radius to use for query")
    parser.add_argument("--cds_timeout", action='store', type=int,
                        default=1200,
                        help="CDS timeout limit")
    parser.add_argument("--chunk_size", action='store', type=int,
                        default=20000,
                        help="CDS query chunk size")

    parser.add_argument("--cds_catalog", action='store', type=str,
                        default='vizier:II/371/des_dr2',
                        help="CDS catalog path to query")
    args = parser.parse_args()
    
    ##make args.search_radius a quantity
    args.search_radius = u.Quantity(args.search_radius)
    
    return args


#############################################################
#############################################################
###main

if __name__ == '__main__':
    args = parse_args()
    targets = load_targets(filename=args.targets,
                           namecol=args.namecol,
                           acol=args.racol,
                           dcol=args.decol)
    xmatch = cds_xmatch(data=targets, racol=args.racol,
                        decol=args.decol,
                        maxsep=args.search_radius,
                        cat2=args.cds_catalog,
                        timeout=args.cds_timeout,
                        column_dict=coldict,
                        chunk_size=args.chunk_size,
                        namecol=args.namecol)
    outfile = '/'.join([args.outdir, args.outfile])
    xmatch.write(outfile)


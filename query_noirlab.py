###query noirlab astrodatalab

import numpy as np, pandas as pd, time, argparse
from dl import queryClient as qc, authClient as ac
from astropy.table import Table, vstack
from astropy import units as u
from io import StringIO



###############################################################
###############################################################
###parameters (maybe make config)

uname = 'yjangordon'
pword = 'Commons_15'
loginfile = '../astrodatalab_login.txt'
testfile = '../data/test_data/test_vlass1.fits'
datafile = '../data/VLASS1QL_components_lite.fits'
vlitecols = ['Component_name', 'RA', 'DEC']

###need to use prefix mydb:// for tables in mydb!
tq = 'select * from mydb://vlass_test'
#tq = 'select top 10 t.ra, t.dec from ls_dr9.tractor_s as t'
#tq = 'select top 10 t.ra, t.dec from mydb:vlass_test as t'

tq2 = ("SELECT * \n"
       + "FROM mydb://vlass_test AS t1\n"
       + "JOIN ls_dr9.tractor_s as t2\n"
       + "ON 1=CONTAINS(POINT('ICRS', t1.ra, t1.dec), CIRCLE('ICRS', t2.ra, t2.dec, 10/3600.0))")
       
tq2 = '''SELECT  o.component_name, g.ls_id,
                (q3c_dist(o.ra,o.dec,g.ra,g.dec)*3600.0) as dist_arcsec
         FROM mydb://vlass_test AS o,
              ls_fr9.tractor_s AS g
         WHERE q3c_join(o.ra, o.dec, g.ra, g.dec, 0.01)'''

tq2 = ("Select t.component_name, g.designation, "
       + "(q3c_dist(t.ra, t.dec, g.ra, g.dec)*3600.0)"
       + "as dist_arcsec FROM mydb://vlass_test AS t, "
       + "gaia_dr3.gaia_source AS g "
       + "WHERE q3c_join(t.ra, t.dec, g.ra, g.dec, 0.003)")

###############################################################
###############################################################
###functions

def login_from_txt(filename):
    'loads username and password from text file of format username\npassword'
    li_info = open(filename).read().split('\n')
    li = {'username': li_info[0], 'password': li_info[1]}
    
    return li


def load_data_to_pandas(filename, litecols=None):
    'load data from file to pandas'
    data = Table.read(filename).to_pandas()
    
    ###ensure bytes decoded
    for col in data.columns:
        if data[col].dtype=='O':
            data[col] = np.array(data[col]).astype(str)
    
    if litecols is not None:
        data = data[litecols]
    
    return data
    

def upload_to_mydb(data, tablename,
                   username=None,
                   password=None):
    'uploads pandas dataframe to astro datalab my_db'
    ###input login details if not provided and login to astro data lab
    if username is None:
        username = input('NOIRLab astro data lab username: ')
    if password is None:
        password = input('NOIRLab astro data lab password: ')
    ac.login(user=username, password=password)
    
    ##upload table
    qc.mydb_import(table=tablename, data=data)
    
    return


def split_large_table_(data, racol='RA', nchunks=4):
    'split a large table into multiple chunks by RA'
    
    rabins = np.linspace(0, 360, nchunks+1)
    chunklist = []
    print(f'splitting data into {nchunks} chunks')
    for i in range(len(rabins)-1):
        a0 = rabins[i]
        a1 = rabins[i+1]
        print(f'   chunk {i+1}: {a0} <= {racol}/deg < {a1}')
        dchunk = data[(data[racol]>=a0) & (data[racol]<a1)]
        chunklist.append(dchunk)
    
    return chunklist


def upload_data_in_chunks(dchunks, table_name, chunknames='_c'):
    'upload data that has been split into chunks to get around upload size limits'
    
    for i in range(len(dchunks)):
        dci = dchunks[i]
        dcname = chunknames.join([table_name, str(i+1)])
        qc.mydb_import(dcname, dci)
        print(f'   uploaded {dcname} to mydb')
    
    return
    


def purge_mydb():
    'purge mydb'
    confirmation = input("WARNING about to purge ALL tables from mydb,\n are you sure? Type 'y' to continue\n ")
    if confirmation in ['y', 'Y', 'yes', 'Yes', 'YES']:
        print('   mydb purge confirmed')
        dblist = qc.mydb_list().split('\n')
        for d in dblist:
            qc.mydb_drop(d)
    
    else:
        print('   mydb purge aborted')
    
    print("List of current mydb contents:\n ")
    print(qc.mydb_list())
    
    return



####designed for vizier, adapt for noirlab -- this is problematic use noirlab xmatch for positional matches
#def write_query(table_to_query='ls_dr9.tractor_s',
#                search_rad=10*u.arcsec, searchprec=0,
#                mydb_table='vlass_test',
#                upra='ra', updec='dec',
#                qra='ra', qdec='dec'):
#    'construct SQL query string from input parameters'
#
#    srad = str(np.round(search_rad.to('arcsec').value, searchprec))
#
##    qstring = f"SELECT * \n FROM mydb://{mydb_table} AS tup \n JOIN {table_to_query} AS db \n ON 1=CONTAINS(POINT('ICRS', tup.{upra}, tup.{updec}), CIRCLE('ICRS', db.{qra}, db.{qdec}, {srad}/3600.))"
#
#    return qstring

###############################################################
###############################################################
###main

login_details = login_from_txt(loginfile)

data = load_data_to_pandas(datafile, litecols=vlitecols)

dsplit = split_large_table_(data=data, racol='RA', nchunks=4)

#upload_data_in_chunks(dchunks=dsplit, table_name='vlass1')

####should also create catalog of ~500k random positions to query for comparison

# This will be the code to generate the cls that will be used in the actual paper. Raw cl's will be exclusively used. Due to the issues with noise in ACT, I will not be using the maps that Marco made and will instead add noise in my own scheme.

import os
import numpy as np
import healpy as hp
from multiprocessing import Pool


nside = 1024
nside_ls = 3*nside

write_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/noiseless_cls/'

try:
    os.mkdir(write_dir)
except Exception:
    pass

conv_dir = '/n/netscratch/dvorkin_lab/Lab/gvalogiannis/maps_cmbl_georgios/maps_cmbl_georgios/'


def file_num(s):
    # Find the index of the first '_'
    start = s.find('_')
    
    # Find the index of the first '.' after the '_'
    end = s.find('.', start)
    
    # Extract and return the substring between '_' and '.'
    return s[start + 1:end]


def fid_func(file):
    num = file_num(file)
    write_file = write_dir + 'fiducial_' + num + '.npy'
    
    map_noiseless = np.load(conv_dir+file, allow_pickle=True).item()['full_sky_noiseless']
    cl = hp.sphtfunc.anafast(map_noiseless)
        
    np.save(write_file, cl)
    return None


if __name__ == "__main__":   
    all_files = np.array(os.listdir(conv_dir))
    all_files = all_files[~np.isin(all_files, os.listdir(write_dir))]
    
    fid_inds = [i for i, s in enumerate(all_files) if s.startswith('f')]
    
    fid_files = all_files[fid_inds]

    with Pool(50) as pool:
        res = pool.map(fid_func, fid_files)
# This will strictly calculate out the cross spectra, not any regular spectra
# All combinations will be probed since this should be easy enough

import os
import numpy as np
import healpy as hp
from multiprocessing import Pool
from scipy.interpolate import interp1d


d_Om = 0.01
d_s8 = 0.015

nside = 1024
nside_ls = 3*nside

write_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_x_wl/raw_cross_cls/'

try:
    os.mkdir(write_dir)
except Exception:
    pass

conv_dir = '/n/netscratch/dvorkin_lab/Lab/gvalogiannis/maps_cmbl_georgios_2/'

cmb_mask_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/masks/'
wl_mask_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_x_wl/wl_masks/'

cmb_survey = 'ACT'
cmb_mask_map = np.load(cmb_mask_dir + cmb_survey + '.npy')
    
wl_survey = 'dr3'
wl_mask_map = np.load(wl_mask_dir + wl_survey + '.npy')


def flip_map(m):
    nside = hp.get_nside(m)
    idx   = np.arange(m.size)
    x, y, z = hp.pix2vec(nside, idx)   # unit vectors
    idx2 = hp.vec2pix(nside, x, y, -z)
    return m[idx2]


def file_num(s, directory=True):
    # Find the index of the last '_', this is right before the number
    start = s.rfind('_')
    if directory:
        return s[start + 1:]
    else:
        end = s.rfind('.')
        return s[start + 1:end]


def healpix_pixel_area(nside):
    n_pixels = hp.nside2npix(nside)
    pixel_area_rad2 = 4 * np.pi / n_pixels  # Area in square radians
    pixel_area_arcmin2 = pixel_area_rad2 * (180 * 60 / np.pi)**2  # Convert to square arcminutes
    return pixel_area_arcmin2


# Example: Calculate pixel area for nside = 64
area_arcmin2 = healpix_pixel_area(nside)
sigma_e = 0.26 
n_eff = 32/6. #gal/arcmin
n_gal_per_pixel = area_arcmin2*n_eff


def make_Euclid_noise():
    nn = np.random.normal(0,sigma_e/np.sqrt(n_gal_per_pixel), 12*nside**2)
    return nn
    
    
def make_ACT_noise():
    noise_ACT = np.loadtxt('/n/home09/kboone/software/Data/Map_Noise/ACT_noise.txt')[:,1]
    noise_ACT = noise_ACT[:np.where(noise_ACT < 1e-9)[0][0]]
    # noise = np.ones(nside_ls)
    # noise[:len(noise_ACT)] = noise_ACT
    nn = hp.synfast(noise_ACT,nside =1024)
    return nn


def deriv_func(map_dir_p):
    flip=False # Don't flip unless we're dealing with a 'f' file
    param = map_dir_p[0:2]
    
    if param == 'Om':
        d_param = d_Om
    elif param == 's8':
        d_param = d_s8
    else:
        return
    
    num = file_num(map_dir_p)
    if 'f' in num:
        flip=True
        map_dir_p = map_dir_p.replace('f', '')
    write_file = write_dir + param + '_' + num + '.npy'
    
    map_dir_m = map_dir_p.replace('_p_', '_m_')
    
    cmb_map_noiseless_p = np.load(conv_dir + map_dir_p + '/cmbl_1024.npy', allow_pickle=True).item()['map']
    wl_maps_p = np.load(conv_dir + map_dir_p + '/kappa_euclid_1024.npy')
    low_z_noiseless_p = wl_maps_p[0]
    high_z_noiseless_p = wl_maps_p[-1]
    
    cmb_map_noiseless_m = np.load(conv_dir + map_dir_m + '/cmbl_1024.npy', allow_pickle=True).item()['map']
    wl_maps_m = np.load(conv_dir + map_dir_m + '/kappa_euclid_1024.npy')
    low_z_noiseless_m = wl_maps_m[0]
    high_z_noiseless_m = wl_maps_m[-1]
    
    if flip:
        cmb_map_noiseless_p = flip_map(cmb_map_noiseless_p)
        low_z_noiseless_p = flip_map(low_z_noiseless_p)
        high_z_noiseless_p = flip_map(high_z_noiseless_p)
        
        cmb_map_noiseless_m = flip_map(cmb_map_noiseless_m)
        low_z_noiseless_m = flip_map(low_z_noiseless_m)
        high_z_noiseless_m = flip_map(high_z_noiseless_m)
    
    data_dict = dict()
    # data_dict = np.load(write_file, allow_pickle=True).item()
    
    nn_wl = make_Euclid_noise()
    nn_cmb = make_ACT_noise()
    # Only unmasked if greater than 1.5
    total_mask = cmb_mask_map + wl_mask_map
    f_map = len(np.where(total_mask > 1.5)[0]) / len(total_mask)
    
    cmb_map_p = cmb_map_noiseless_p + nn_cmb
    if cmb_survey == 'ACT':
        alm_p = hp.map2alm(cmb_map_p, lmax=2090) # This is the max ell I had access to.
        cmb_map_p = hp.alm2map(alm_p, nside)
    if cmb_survey == 'Planck':
        cmb_map_p = hp.sphtfunc.smoothing(cmb_map_p, fwhm=fwhm_rad)
    low_z_p = low_z_noiseless_p + nn_wl
    high_z_p = high_z_noiseless_p + nn_wl

    cmb_map_m = cmb_map_noiseless_m + nn_cmb
    if cmb_survey == 'ACT':
        alm_m = hp.map2alm(cmb_map_m, lmax=2090) # This is the max ell I had access to.
        cmb_map_m = hp.alm2map(alm_m, nside)
    if cmb_survey == 'Planck':
        cmb_map_m = hp.sphtfunc.smoothing(cmb_map_m, fwhm=fwhm_rad)
    low_z_m = low_z_noiseless_m + nn_wl
    high_z_m = high_z_noiseless_m + nn_wl

    cmb_map_p[np.where(total_mask < 1.5)] = 0
    low_z_p[np.where(total_mask < 1.5)] = 0
    high_z_p[np.where(total_mask < 1.5)] = 0

    cmb_map_m[np.where(total_mask < 1.5)] = 0
    low_z_m[np.where(total_mask < 1.5)] = 0
    high_z_m[np.where(total_mask < 1.5)] = 0

    cmb_cross_low_p = hp.sphtfunc.anafast(cmb_map_p, map2=low_z_p) / f_map
    cmb_cross_high_p = hp.sphtfunc.anafast(cmb_map_p, map2=high_z_p) / f_map

    cmb_cross_low_m = hp.sphtfunc.anafast(cmb_map_m, map2=low_z_m) / f_map
    cmb_cross_high_m = hp.sphtfunc.anafast(cmb_map_m, map2=high_z_m) / f_map

    data_dict[cmb_survey + '_x_' + wl_survey + '_low_z'] = (cmb_cross_low_p - cmb_cross_low_m) / (2*d_param)
    data_dict[cmb_survey + '_x_' + wl_survey + '_high_z'] = (cmb_cross_high_p - cmb_cross_high_m) / (2*d_param)
            
    np.save(write_file, data_dict)
    return None


if __name__ == "__main__":   
    all_dirs = np.array(os.listdir(conv_dir))
    
    Om_inds = [i for i, s in enumerate(all_dirs) if s.startswith('Om_p')]
    s8_inds = [i for i, s in enumerate(all_dirs) if s.startswith('s8_p')]
    
    Om_dirs = all_dirs[Om_inds]
    s8_dirs = all_dirs[s8_inds]
    
    # Add in the flipped "directories"
    Om_f_dirs = []
    s8_f_dirs = []
    for Om_dir in Om_dirs:
        Om_num = file_num(Om_dir)
        Om_f_dirs.append(Om_dir.replace(Om_num, Om_num+'f'))
    for s8_dir in s8_dirs:
        s8_num = file_num(s8_dir)
        s8_f_dirs.append(s8_dir.replace(s8_num, s8_num+'f'))
    Om_dirs = np.append(Om_dirs, np.array(Om_f_dirs))
    s8_dirs = np.append(s8_dirs, np.array(s8_f_dirs))
    
    Om_nums = []
    s8_nums = []
    for Om_dir in Om_dirs:
        Om_nums.append(file_num(Om_dir))
    for s8_dir in s8_dirs:
        s8_nums.append(file_num(s8_dir))
    Om_nums = np.array(Om_nums)
    s8_nums = np.array(s8_nums)  
    
    written_files = np.array(os.listdir(write_dir))
    
    Om_inds_written = [i for i, s in enumerate(written_files) if s.startswith('O')]
    s8_inds_written = [i for i, s in enumerate(written_files) if s.startswith('s')]
    
    Om_files = written_files[Om_inds_written]
    s8_files = written_files[s8_inds_written]
    
    Om_nums_written = []
    s8_nums_written = []
    for Om_file in Om_files:
        Om_nums_written.append(file_num(Om_file, directory=False))
    for s8_file in s8_files:
        s8_nums_written.append(file_num(s8_file, directory=False))
    Om_nums_written = np.array(Om_nums_written)
    s8_nums_written = np.array(s8_nums_written)
    
    process_dirs = np.append(Om_dirs[~np.isin(Om_nums, Om_nums_written)], s8_dirs[~np.isin(s8_nums, s8_nums_written)])
    
    # process_dirs = process_dirs[:16]
    
    print(process_dirs)

    with Pool(16) as pool:
        res = pool.map(deriv_func, process_dirs)

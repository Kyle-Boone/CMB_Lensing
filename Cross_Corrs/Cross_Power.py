import os
import numpy as np
import healpy as hp
import pymaster as nmt
from multiprocessing import Pool
from scipy.interpolate import interp1d

cmb_survey = 'Planck'
wl_survey = 'dr3'

d_Om = 0.01
d_s8 = 0.015

nside = 1024
nside_ls = 3072

lmax = 2000
ell_ini = np.append(np.arange(lmax-1)+2, np.array([lmax+1]))
ell_end = np.append(np.arange(lmax-1)+3, np.array([nside_ls]))
b = nmt.NmtBin.from_edges(ell_ini, ell_end)
l = b.get_effective_ells()

write_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_x_wl/cl_lmax' + str(lmax) + '_' + cmb_survey + '_' + wl_survey + '/'

try:
    os.mkdir(write_dir)
except Exception:
    pass

conv_dir = '/n/netscratch/dvorkin_lab/Lab/gvalogiannis/maps_cmbl_georgios_2/'

cmb_mask_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/masks/'
wl_mask_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_x_wl/wl_masks/'

cmb_mask_map = np.load(cmb_mask_dir + cmb_survey + '.npy')
wl_mask_map = np.load(wl_mask_dir + wl_survey + '.npy')

cmb_mask = nmt.mask_apodization(cmb_mask_map, 0.25, apotype="Smooth")
wl_mask = nmt.mask_apodization(wl_mask_map, 0.25, apotype="Smooth")


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
    
    
def make_CMB_noise():
    if cmb_survey == 'Planck':
        N = 1e7
        x = [10,75.32337589203746	,160.85999889800618	,340.4967383339498	,871.2051805682871	,1724.809488637684	]
        y = [1,1.0182,1.1348,1.4041,1.5198,2.5557	]
        f = interp1d(x, y, kind='linear', fill_value="extrapolate")
        cmb_lensing_noise = hp.synfast(2*f(np.arange(3000))/N,nside =1024)
        return cmb_lensing_noise
    else:
        return None
    
    
def file_num(s, directory=True):
    # Find the index of the last '_', this is right before the number
    start = s.rfind('_')
    if directory:
        return s[start + 1:]
    else:
        end = s.rfind('.')
        return s[start + 1:end]


def fid_func(map_dir):
    num = file_num(map_dir)
    write_file = write_dir + 'fiducial_' + num + '.npy'
    
    cmb_map_noiseless = np.load(conv_dir + map_dir + '/cmbl_1024.npy', allow_pickle=True).item()['map']
    wl_maps = np.load(conv_dir + map_dir + '/kappa_euclid_1024.npy')
    low_z_noiseless = wl_maps[0]
    high_z_noiseless = wl_maps[-1]
    
    nn_cmb = make_CMB_noise()
    nn_wl = make_Euclid_noise()
    
    cmb_map = cmb_map_noiseless + nn_cmb
    if cmb_survey == 'Planck':
        # smooth Planck - 
        fwhm_arcmin = 10  # FWHM in arcminutes
        fwhm_rad = np.radians(fwhm_arcmin / 60.0)  # Convert to radians
        cmb_map = hp.sphtfunc.smoothing(cmb_map, fwhm=fwhm_rad)
    low_z = low_z_noiseless + nn_wl
    high_z = high_z_noiseless + nn_wl
    
    f_0_cmb = nmt.NmtField(cmb_mask, [cmb_map])
    f_0_low = nmt.NmtField(wl_mask, [low_z])
    f_0_high = nmt.NmtField(wl_mask, [high_z])
    
    cl_cmb = nmt.compute_full_master(f_0_cmb, f_0_cmb, b)
    cl_low = nmt.compute_full_master(f_0_low, f_0_low, b)
    cl_high = nmt.compute_full_master(f_0_high, f_0_high, b)
    cl_cmb_x_low = nmt.compute_full_master(f_0_cmb, f_0_low, b)
    cl_cmb_x_high = nmt.compute_full_master(f_0_cmb, f_0_high, b)
    
    data_dict = dict()
    data_dict['l'] = l[:-1]
    data_dict['cmb'] = cl_cmb[0][:-1]
    data_dict['low_z'] = cl_low[0][:-1]
    data_dict['high_z'] = cl_high[0][:-1]
    data_dict['cmb_x_low'] = cl_cmb_x_low[0][:-1]
    data_dict['cmb_x_high'] = cl_cmb_x_high[0][:-1]
    
    np.save(write_file, data_dict)
    return None


def deriv_func(map_dir_p):
    param = map_dir_p[0:2]
    
    if param == 'Om':
        d_param = d_Om
    elif param == 's8':
        d_param = d_s8
    else:
        return
    
    num = file_num(map_dir_p)
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
    
    nn_cmb = make_CMB_noise()
    nn_wl = make_Euclid_noise()
    
    cmb_map_p = cmb_map_noiseless_p + nn_cmb
    if cmb_survey == 'Planck':
        # smooth Planck - 
        fwhm_arcmin = 10  # FWHM in arcminutes
        fwhm_rad = np.radians(fwhm_arcmin / 60.0)  # Convert to radians
        cmb_map_p = hp.sphtfunc.smoothing(cmb_map_p, fwhm=fwhm_rad)
    low_z_p = low_z_noiseless_p + nn_wl
    high_z_p = high_z_noiseless_p + nn_wl
    
    cmb_map_m = cmb_map_noiseless_m + nn_cmb
    if cmb_survey == 'Planck':
        # smooth Planck - 
        fwhm_arcmin = 10  # FWHM in arcminutes
        fwhm_rad = np.radians(fwhm_arcmin / 60.0)  # Convert to radians
        cmb_map_m = hp.sphtfunc.smoothing(cmb_map_m, fwhm=fwhm_rad)
    low_z_m = low_z_noiseless_m + nn_wl
    high_z_m = high_z_noiseless_m + nn_wl
    
    f_0_cmb_p = nmt.NmtField(cmb_mask, [cmb_map_p])
    f_0_low_p = nmt.NmtField(wl_mask, [low_z_p])
    f_0_high_p = nmt.NmtField(wl_mask, [high_z_p])
    
    f_0_cmb_m = nmt.NmtField(cmb_mask, [cmb_map_m])
    f_0_low_m = nmt.NmtField(wl_mask, [low_z_m])
    f_0_high_m = nmt.NmtField(wl_mask, [high_z_m])
    
    cl_cmb_p = nmt.compute_full_master(f_0_cmb_p, f_0_cmb_p, b)
    cl_low_p = nmt.compute_full_master(f_0_low_p, f_0_low_p, b)
    cl_high_p = nmt.compute_full_master(f_0_high_p, f_0_high_p, b)
    cl_cmb_x_low_p = nmt.compute_full_master(f_0_cmb_p, f_0_low_p, b)
    cl_cmb_x_high_p = nmt.compute_full_master(f_0_cmb_p, f_0_high_p, b)
    
    cl_cmb_m = nmt.compute_full_master(f_0_cmb_m, f_0_cmb_m, b)
    cl_low_m = nmt.compute_full_master(f_0_low_m, f_0_low_m, b)
    cl_high_m = nmt.compute_full_master(f_0_high_m, f_0_high_m, b)
    cl_cmb_x_low_m = nmt.compute_full_master(f_0_cmb_m, f_0_low_m, b)
    cl_cmb_x_high_m = nmt.compute_full_master(f_0_cmb_m, f_0_high_m, b)
    
    data_dict = dict()
    data_dict['l'] = l[:-1]
    data_dict['cmb'] = (cl_cmb_p[0][:-1]-cl_cmb_m[0][:-1])/(2*d_param)
    data_dict['low_z'] = (cl_low_p[0][:-1]-cl_low_m[0][:-1])/(2*d_param)
    data_dict['high_z'] = (cl_high_p[0][:-1]-cl_high_m[0][:-1])/(2*d_param)
    data_dict['cmb_x_low'] = (cl_cmb_x_low_p[0][:-1]-cl_cmb_x_low_m[0][:-1])/(2*d_param)
    data_dict['cmb_x_high'] = (cl_cmb_x_high_p[0][:-1]-cl_cmb_x_high_m[0][:-1])/(2*d_param)
    
    np.save(write_file, data_dict)
    return None


def process_dir(map_dir):
    if map_dir.startswith('f'):
        fid_func(map_dir)
    elif map_dir.startswith('O'):
        deriv_func(map_dir)
    elif map_dir.startswith('s'):
        deriv_func(map_dir)
    return None


if __name__ == "__main__":   
    all_dirs = np.array(os.listdir(conv_dir))
    
    fid_inds = [i for i, s in enumerate(all_dirs) if s.startswith('f')]
    Om_inds = [i for i, s in enumerate(all_dirs) if s.startswith('Om_p')]
    s8_inds = [i for i, s in enumerate(all_dirs) if s.startswith('s8_p')]
    
    fid_dirs = all_dirs[fid_inds]
    Om_dirs = all_dirs[Om_inds]
    s8_dirs = all_dirs[s8_inds]
    
    fid_nums = []
    Om_nums = []
    s8_nums = []
    for fid_dir in fid_dirs:
        fid_nums.append(file_num(fid_dir))
    for Om_dir in Om_dirs:
        Om_nums.append(file_num(Om_dir))
    for s8_dir in s8_dirs:
        s8_nums.append(file_num(s8_dir))
    fid_nums = np.array(fid_nums)
    Om_nums = np.array(Om_nums)
    s8_nums = np.array(s8_nums)  
    
    written_files = np.array(os.listdir(write_dir))
    
    fid_inds_written = [i for i, s in enumerate(written_files) if s.startswith('f')]
    Om_inds_written = [i for i, s in enumerate(written_files) if s.startswith('O')]
    s8_inds_written = [i for i, s in enumerate(written_files) if s.startswith('s')]
    
    fid_files = written_files[fid_inds_written]
    Om_files = written_files[Om_inds_written]
    s8_files = written_files[s8_inds_written]
    
    fid_nums_written = []
    Om_nums_written = []
    s8_nums_written = []
    for fid_file in fid_files:
        fid_nums_written.append(file_num(fid_file, directory=False))
    for Om_file in Om_files:
        Om_nums_written.append(file_num(Om_file, directory=False))
    for s8_file in s8_files:
        s8_nums_written.append(file_num(s8_file, directory=False))
    fid_nums_written = np.array(fid_nums_written)
    Om_nums_written = np.array(Om_nums_written)
    s8_nums_written = np.array(s8_nums_written)
    
    process_dirs = np.append(np.append(fid_dirs[~np.isin(fid_nums, fid_nums_written)], Om_dirs[~np.isin(Om_nums, Om_nums_written)]), s8_dirs[~np.isin(s8_nums, s8_nums_written)])

    with Pool(32) as pool:
        res = pool.map(process_dir, process_dirs)
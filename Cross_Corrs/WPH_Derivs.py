import os
import time
import torch
import pywph as pw
import numpy as np
import healpy as hp
from multiprocessing import Pool
from scipy.interpolate import interp1d


# WST Hyperparameters
M = 2048
N = M
J = 10
L = 4
dn = 0
j_min = 0

# Directory to write data to
cmb_survey = 'SPT'
wl_survey = 'dr3'
comb_survey = cmb_survey + '_x_' + wl_survey
write_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_x_wl/'+comb_survey+'_one_patch_wph_j'+str(J)+'_l'+str(L)+'/'

try:
    os.mkdir(write_dir)
except Exception:
    pass

start = time.time()

wph_op = pw.WPHOp(M, N, J, L=L,j_min=j_min, dn=dn, device='cpu')

end = time.time()
print(end - start)

# Cosmology Parameters
Om = 0.3175
d_Om = 0.01

s8 = 0.834
d_s8 = 0.015

# Other params
nside = 1024
nside_ls = 3*nside

# Where data is
conv_dir = '/n/netscratch/dvorkin_lab/Lab/gvalogiannis/maps_cmbl_georgios_2/'
cmb_mask_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/masks/'
wl_mask_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_x_wl/wl_masks/'
patch_dir = '/n/home09/kboone/software/Data/Patches/'

cmb_mask_map = np.load(cmb_mask_dir + cmb_survey + '.npy')
wl_mask_map = np.load(wl_mask_dir + wl_survey + '.npy')

total_mask = cmb_mask_map + wl_mask_map + 1
    
patch_inds = np.load(patch_dir + 'one_patch_nside_1024_xsize_2048.npy')
unmask = patch_inds > -0.5

SPT_surveys = ['SPT_main', 'SPT_summer']

SPT_mask_maps = []
for survey in SPT_surveys:
    SPT_mask_maps.append(np.load(cmb_mask_dir + survey + '.npy'))
    
SPT_weights = SPT_mask_maps[0]+SPT_mask_maps[1]


def WPH_maps(cmb, lowz, highz):
    cmb_patches = np.zeros_like(patch_inds).astype(float)
    cmb_patches[unmask] = np.take(cmb, patch_inds[unmask])
    
    lowz_patches = np.zeros_like(patch_inds).astype(float)
    lowz_patches[unmask] = np.take(lowz, patch_inds[unmask])
    
    highz_patches = np.zeros_like(patch_inds).astype(float)
    highz_patches[unmask] = np.take(highz, patch_inds[unmask])
    
    for i in np.arange(len(cmb_patches)):
        wph_cmb_low = wph_op([cmb_patches[i], lowz_patches[i]], cross=True, ret_wph_obj=True)
        wph_cmb_low.to_isopar()
        
        wph_cmb_high = wph_op([cmb_patches[i], highz_patches[i]], cross=True, ret_wph_obj=True)
        wph_cmb_high.to_isopar()
        
        wph_low_cmb = wph_op([lowz_patches[i], cmb_patches[i]], cross=True, ret_wph_obj=True)
        wph_low_cmb.to_isopar()
        
        wph_high_cmb = wph_op([highz_patches[i], cmb_patches[i]], cross=True, ret_wph_obj=True)
        wph_high_cmb.to_isopar()
        
        s00_cmb_low, _ = wph_cmb_low.get_coeffs("S00")
        s11_cmb_low, _ = wph_cmb_low.get_coeffs("S11")
        s01_cmb_low, _ = wph_cmb_low.get_coeffs("S01")
        c01_cmb_low, _ = wph_cmb_low.get_coeffs("C01")
        
        s00_cmb_high, _ = wph_cmb_high.get_coeffs("S00")
        s11_cmb_high, _ = wph_cmb_high.get_coeffs("S11")
        s01_cmb_high, _ = wph_cmb_high.get_coeffs("S01")
        c01_cmb_high, _ = wph_cmb_high.get_coeffs("C01")
        
        s01_low_cmb, _ = wph_low_cmb.get_coeffs("S01")
        c01_low_cmb, _ = wph_low_cmb.get_coeffs("C01")
        
        s01_high_cmb, _ = wph_high_cmb.get_coeffs("S01")
        c01_high_cmb, _ = wph_high_cmb.get_coeffs("C01")
        
        if i == 0:
            data_vecs = np.atleast_2d(np.concatenate([s00_cmb_low, s11_cmb_low, s01_cmb_low, c01_cmb_low, s00_cmb_high, s11_cmb_high, s01_cmb_high, c01_cmb_high, s01_low_cmb, c01_low_cmb, s01_high_cmb, c01_high_cmb]))
        else:
            data_vecs = np.append(data_vecs, np.atleast_2d(np.concatenate([s00_cmb_low, s11_cmb_low, s01_cmb_low, c01_cmb_low, s00_cmb_high, s11_cmb_high, s01_cmb_high, c01_cmb_high, s01_low_cmb, c01_low_cmb, s01_high_cmb, c01_high_cmb])), axis=0)
    
    data_vec = np.sum(data_vecs, axis=0)
    return data_vec
        

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


def make_SPT_noise():
    N = 1e7
    # https://arxiv.org/pdf/2403.17925
    x = [34.150734230131775	,102.66510443231208	,256.3784422342604	,653.0266368808517	,1410.2304370377647	,2286.7839428151624	,3047.951172608016	,3301.4963574077037	,3827.9125881096406	,4746.201775079477	]
    y = [0.0639769792365802,0.07335548408566443,0.09262018291845851,0.12097806351013743,0.18307410239368344,0.2754037235130334,0.5975503865801056,0.7942895129566045,0.6526302418239897,1.7343096186104023]
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    main_noise = hp.synfast(f(np.arange(nside_ls))/N,nside =1024)
    
    x = [35.190688373236284	,50.790359333427915	,89.97065841891472	,197.08472778011262	,385.2789162003061	,1006.4549748097186	,2181.7056856609784	,2784.791530071566	,3222.369693949373	,3725.7883975549175	,4298.874152276615	]
    y = [0.07856998685838304,0.08443030118214156,0.10849544565221234,0.164707182470824,0.24008135934943148,0.3648854446906422,0.5788268536804575,1.0909730047081443,1.659722689685696,1.3690446452863363,1.9974160387048998]
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    summer_noise = hp.synfast(f(np.arange(nside_ls))/N,nside =1024)
    
    total_noise = np.zeros_like(main_noise)
    total_noise[np.where(SPT_mask_maps[0] > 0.5)] += main_noise[np.where(SPT_mask_maps[0] > 0.5)]
    total_noise[np.where(SPT_mask_maps[1] > 0.5)] += summer_noise[np.where(SPT_mask_maps[1] > 0.5)]
    total_noise[np.where(SPT_weights > 0.5)] = total_noise[np.where(SPT_weights > 0.5)] / SPT_weights[np.where(SPT_weights > 0.5)]
    
    return total_noise


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
    
    nn_cmb = make_SPT_noise()
    nn_wl = make_Euclid_noise()
    
    cmb_p = cmb_map_noiseless_p + nn_cmb
    if cmb_survey == 'ACT':
        alm_p = hp.map2alm(cmb_p, lmax=2090) # This is the max ell I had access to.
        cmb_p = hp.alm2map(alm_p, nside)
    if cmb_survey == 'Planck':
        cmb_p = hp.sphtfunc.smoothing(cmb_p, fwhm=fwhm_rad)
    low_z_p = low_z_noiseless_p + nn_wl
    high_z_p = high_z_noiseless_p + nn_wl
    
    cmb_m = cmb_map_noiseless_m + nn_cmb
    if cmb_survey == 'ACT':
        alm_m = hp.map2alm(cmb_m, lmax=2090) # This is the max ell I had access to.
        cmb_m = hp.alm2map(alm_m, nside)
    if cmb_survey == 'Planck':
        cmb_m = hp.sphtfunc.smoothing(cmb_m, fwhm=fwhm_rad)
    low_z_m = low_z_noiseless_m + nn_wl
    high_z_m = high_z_noiseless_m + nn_wl

    cmb_p[np.where(total_mask < 2.5)] = 0
    low_z_p[np.where(total_mask < 2.5)] = 0
    high_z_p[np.where(total_mask < 2.5)] = 0

    cmb_m[np.where(total_mask < 2.5)] = 0
    low_z_m[np.where(total_mask < 2.5)] = 0
    high_z_m[np.where(total_mask < 2.5)] = 0
    
    wph_p = WPH_maps(cmb_p, low_z_p, high_z_p)
    wph_m = WPH_maps(cmb_m, low_z_m, high_z_m)
    
    dwph = (wph_p - wph_m) / (2*d_param)
            
    np.save(write_file, dwph)
    return None


if __name__ == "__main__":  
    all_dirs = np.array(os.listdir(conv_dir))
    
    Om_inds = [i for i, s in enumerate(all_dirs) if s.startswith('Om_p')]
    s8_inds = [i for i, s in enumerate(all_dirs) if s.startswith('s8_p')]
    
    Om_dirs = all_dirs[Om_inds]
    s8_dirs = all_dirs[s8_inds]
    
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
    
    with Pool(14) as pool:
        res = pool.map(deriv_func, process_dirs)

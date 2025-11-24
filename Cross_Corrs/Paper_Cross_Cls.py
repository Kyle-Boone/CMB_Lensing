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

cmb_mask_files = os.listdir(cmb_mask_dir)
cmb_surveys = [cmb_mask_file.rstrip("npy") for cmb_mask_file in cmb_mask_files]
cmb_surveys = [cmb_survey.rstrip(".") for cmb_survey in cmb_surveys]
# Don't want SPT_main or SPT_summer
cmb_surveys = cmb_surveys[:-2]

cmb_mask_maps = []
for cmb_survey in cmb_surveys:
    cmb_mask_maps.append(np.load(cmb_mask_dir + cmb_survey + '.npy'))
    
wl_mask_files = os.listdir(wl_mask_dir)
wl_surveys = [wl_mask_file.rstrip("npy") for wl_mask_file in wl_mask_files]
wl_surveys = [wl_survey.rstrip(".") for wl_survey in wl_surveys]

wl_mask_maps = []
for wl_survey in wl_surveys:
    wl_mask_maps.append(np.load(wl_mask_dir + wl_survey + '.npy'))
    
SPT_surveys = ['SPT_main', 'SPT_summer']

SPT_mask_maps = []
for survey in SPT_surveys:
    SPT_mask_maps.append(np.load(cmb_mask_dir + survey + '.npy'))
    
SPT_weights = SPT_mask_maps[0]+SPT_mask_maps[1]

# smooth Planck - 
fwhm_arcmin = 10  # FWHM in arcminutes
fwhm_rad = np.radians(fwhm_arcmin / 60.0)  # Convert to radians


def file_num(s, directory=True):
    # Find the index of the last '_', this is right before the number
    start = s.rfind('_')
    if directory:
        return s[start + 1:]
    else:
        end = s.rfind('.')
        return s[start + 1:end]
    
    
def antipode_map(m):
    nside = hp.get_nside(m)
    idx   = np.arange(m.size)
    x, y, z = hp.pix2vec(nside, idx)   # unit vectors
    idx2 = hp.vec2pix(nside, -x, -y, -z)
    return m[idx2]


def flip_map(m):
    nside = hp.get_nside(m)
    idx   = np.arange(m.size)
    x, y, z = hp.pix2vec(nside, idx)   # unit vectors
    idx2 = hp.vec2pix(nside, x, y, -z)
    return m[idx2]


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
    
    
def make_CMB_noise(survey_ind):
    survey = cmb_surveys[survey_ind]
    if survey == 'Planck':
        N = 1e7
        x = [10,75.32337589203746	,160.85999889800618	,340.4967383339498	,871.2051805682871	,1724.809488637684	]
        y = [1,1.0182,1.1348,1.4041,1.5198,2.5557	]
        f = interp1d(x, y, kind='linear', fill_value="extrapolate")
        nn = hp.synfast(2*f(np.arange(nside_ls))/N,nside =1024)
        return nn
    elif survey == 'SO':
        N = 1e7
        x = [34.19233078337337	,71.86940802505276	,127.50974815233886	,208.08224631676606	,335.80966800506377	,626.9324561102427	,1193.7167598928752	,2274.5883052693794	,2734.7612461893104	,3075.370469000327	,4235.061816130342	]
        y = [0.1637325958704072,0.1658702835721177,0.19384826328334373,0.2923170686360547,0.38945080711492985,0.4148250745921913,0.5015602317089521,0.6742839831054106,1.1265018636427957,1.663103233757362,1.990063198637278]
        f = interp1d(x, y, kind='linear', fill_value="extrapolate")
        nn = hp.synfast(f(np.arange(nside_ls))/N,nside =1024)
        return nn
    elif survey == 'ACT':
        noise_ACT = np.loadtxt('/n/home09/kboone/software/Data/Map_Noise/ACT_noise.txt')[:,1]
        noise_ACT = noise_ACT[:np.where(noise_ACT < 1e-9)[0][0]]
        # noise = np.ones(nside_ls)
        # noise[:len(noise_ACT)] = noise_ACT
        nn = hp.synfast(noise_ACT,nside =1024)
        return nn
    elif survey == 'SPT':
        nn = make_SPT_noise()
        return nn
    return None
    
    
def fid_func(map_dir):
    # I will mask based on the intersection of the two masks involved
    num = file_num(map_dir)
    # Hardcoded, issues with this file currently
    if num == '773':
        return None
    write_file = write_dir + 'fiducial_' + num + '.npy'
    
    cmb_map_noiseless = np.load(conv_dir + map_dir + '/cmbl_1024.npy', allow_pickle=True).item()['map']
    wl_maps = np.load(conv_dir + map_dir + '/kappa_euclid_1024.npy')
    low_z_noiseless = wl_maps[0]
    high_z_noiseless = wl_maps[-1]
    
    data_dict = dict()
    # data_dict = np.load(write_file, allow_pickle=True).item()
    
    nn_wl = make_Euclid_noise()
    for cmb_survey_ind in np.arange(len(cmb_surveys)):
        nn_cmb = make_CMB_noise(cmb_survey_ind)
        for wl_survey_ind in np.arange(len(wl_surveys)):
            cmb_survey = cmb_surveys[cmb_survey_ind]
            wl_survey = wl_surveys[wl_survey_ind]
            # Only unmasked if greater than 1.5
            total_mask = cmb_mask_maps[cmb_survey_ind] + wl_mask_maps[wl_survey_ind]
            f_map = len(np.where(total_mask > 1.5)[0]) / len(total_mask)
            
            cmb_map = cmb_map_noiseless + nn_cmb
            if cmb_survey == 'ACT':
                alm = hp.map2alm(cmb_map, lmax=2090) # This is the max ell I had access to.
                cmb_map = hp.alm2map(alm, nside)
            if cmb_survey == 'Planck':
                cmb_map = hp.sphtfunc.smoothing(cmb_map, fwhm=fwhm_rad)
            
            low_z = low_z_noiseless + nn_wl
            high_z = high_z_noiseless + nn_wl
            
            cmb_map[np.where(total_mask < 1.5)] = 0
            low_z[np.where(total_mask < 1.5)] = 0
            high_z[np.where(total_mask < 1.5)] = 0
            
            cmb_cross_low = hp.sphtfunc.anafast(cmb_map, map2=low_z) / f_map
            cmb_cross_high = hp.sphtfunc.anafast(cmb_map, map2=high_z) / f_map
            
            data_dict[cmb_survey + '_x_' + wl_survey + '_low_z'] = cmb_cross_low
            data_dict[cmb_survey + '_x_' + wl_survey + '_high_z'] = cmb_cross_high
            
    np.save(write_file, data_dict)
    return None


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
    for cmb_survey_ind in np.arange(len(cmb_surveys)):
        nn_cmb = make_CMB_noise(cmb_survey_ind)
        for wl_survey_ind in np.arange(len(wl_surveys)):
            cmb_survey = cmb_surveys[cmb_survey_ind]
            wl_survey = wl_surveys[wl_survey_ind]
            # Only unmasked if greater than 1.5
            total_mask = cmb_mask_maps[cmb_survey_ind] + wl_mask_maps[wl_survey_ind]
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
    
    # Add in the flipped "directories"
    # Om_f_dirs = []
    # s8_f_dirs = []
    # for Om_dir in Om_dirs:
    #     Om_num = file_num(Om_dir)
    #     Om_f_dirs.append(Om_dir.replace(Om_num, Om_num+'f'))
    # for s8_dir in s8_dirs:
    #     s8_num = file_num(s8_dir)
    #     s8_f_dirs.append(s8_dir.replace(s8_num, s8_num+'f'))
    # Om_dirs = np.append(Om_dirs, np.array(Om_f_dirs))
    # s8_dirs = np.append(s8_dirs, np.array(s8_f_dirs))
    
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
        fid_nums_written.append('773')
    for Om_file in Om_files:
        Om_nums_written.append(file_num(Om_file, directory=False))
    for s8_file in s8_files:
        s8_nums_written.append(file_num(s8_file, directory=False))
    fid_nums_written = np.array(fid_nums_written)
    Om_nums_written = np.array(Om_nums_written)
    s8_nums_written = np.array(s8_nums_written)
    
    process_dirs = np.append(np.append(fid_dirs[~np.isin(fid_nums, fid_nums_written)], Om_dirs[~np.isin(Om_nums, Om_nums_written)]), s8_dirs[~np.isin(s8_nums, s8_nums_written)])
    
    # process_dirs = np.append(np.append(fid_dirs, Om_dirs), s8_dirs)

    with Pool(50) as pool: # Used all 400 GB for 4 hours
        res = pool.map(process_dir, process_dirs)

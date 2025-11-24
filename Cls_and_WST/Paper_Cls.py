# This will be the code to generate the cls that will be used in the actual paper. Raw cl's will be exclusively used. Due to the issues with noise in ACT, I will not be using the maps that Marco made and will instead add noise in my own scheme.

import os
import numpy as np
import healpy as hp
from multiprocessing import Pool
from scipy.interpolate import interp1d


d_Om = 0.01
d_s8 = 0.015

nside = 1024
nside_ls = 3*nside

write_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/cls/'

try:
    os.mkdir(write_dir)
except Exception:
    pass

conv_dir = '/n/netscratch/dvorkin_lab/Lab/gvalogiannis/maps_cmbl_georgios/maps_cmbl_georgios/'
mask_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/masks/'

mask_files = os.listdir(mask_dir)
surveys = [mask_file.rstrip("npy") for mask_file in mask_files]
surveys = [survey.rstrip(".") for survey in surveys]
# Don't want SPT_main or SPT_summer
surveys = surveys[0:4]

mask_maps = []
for survey in surveys:
    mask_maps.append(np.load(mask_dir + survey + '.npy'))
    
f_maps = [] # The fraction of the sky that is unmasked, gives a first order correction to raw Cl's which will be applied here.
for mask_map in mask_maps:
    f_maps.append(len(np.where(mask_map > 0.5)[0]) / len(mask_map))
    
SPT_surveys = ['SPT_main', 'SPT_summer']

SPT_mask_maps = []
for survey in SPT_surveys:
    SPT_mask_maps.append(np.load(mask_dir + survey + '.npy'))
    
SPT_weights = SPT_mask_maps[0]+SPT_mask_maps[1]

# smooth Planck - 
fwhm_arcmin = 10  # FWHM in arcminutes
fwhm_rad = np.radians(fwhm_arcmin / 60.0)  # Convert to radians


def file_num(s):
    # Find the index of the first '_'
    start = s.find('_')
    
    # Find the index of the first '.' after the '_'
    end = s.find('.', start)
    
    # Extract and return the substring between '_' and '.'
    return s[start + 1:end]


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
    survey = surveys[survey_ind]
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


def fid_func(file):
    num = file_num(file)
    write_file = write_dir + 'fiducial_' + num + '.npy'
    
    map_noiseless = np.load(conv_dir+file, allow_pickle=True).item()['full_sky_noiseless']
    
    data_dict = dict()
    
    for survey_ind in np.arange(len(surveys)):
        survey = surveys[survey_ind]
        nn = make_CMB_noise(survey_ind)
        
        cmb_map = map_noiseless + nn
        if survey == 'ACT':
            alm = hp.map2alm(cmb_map, lmax=2090) # This is the max ell I had access to.
            cmb_map = hp.alm2map(alm, nside)
        if survey == 'Planck':
            cmb_map = hp.sphtfunc.smoothing(cmb_map, fwhm=fwhm_rad)
            
        f_map = f_maps[survey_ind]
        cmb_map[np.where(mask_maps[survey_ind] < 0.5)] = 0
        cl = hp.sphtfunc.anafast(cmb_map) / f_map
        
        data_dict[survey] = cl
        
    np.save(write_file, data_dict)
    return None


def deriv_func(file):
    num = file_num(file)
    param = file[0:2]
    
    if param == 'Om':
        d_param = d_Om
    elif param == 's8':
        d_param = d_s8
    else:
        return None
    
    write_file = write_dir + param + '_' + num + '.npy'
    
    data = np.load(conv_dir+file, allow_pickle=True)
    map_p_noiseless = data[0]['full_sky_noiseless']
    map_m_noiseless = data[1]['full_sky_noiseless']
    
    data_dict = dict()
    
    for survey_ind in np.arange(len(surveys)):
        survey = surveys[survey_ind]
        nn = make_CMB_noise(survey_ind)
        
        cmb_map_p = map_p_noiseless + nn
        cmb_map_m = map_m_noiseless + nn
        
        if survey == 'ACT':
            alm_p = hp.map2alm(cmb_map_p, lmax=2090) # This is the max ell I had access to.
            cmb_map_p = hp.alm2map(alm_p, nside)
            alm_m = hp.map2alm(cmb_map_m, lmax=2090) # This is the max ell I had access to.
            cmb_map_m = hp.alm2map(alm_m, nside)
        if survey == 'Planck':
            cmb_map_p = hp.sphtfunc.smoothing(cmb_map_p, fwhm=fwhm_rad)
            cmb_map_m = hp.sphtfunc.smoothing(cmb_map_m, fwhm=fwhm_rad)
            
        f_map = f_maps[survey_ind]
        cmb_map_p[np.where(mask_maps[survey_ind] < 0.5)] = 0
        cmb_map_m[np.where(mask_maps[survey_ind] < 0.5)] = 0
        cl_p = hp.sphtfunc.anafast(cmb_map_p) / f_map
        cl_m = hp.sphtfunc.anafast(cmb_map_m) / f_map
        
        data_dict[survey] = (cl_p - cl_m) / (2*d_param)
        
    np.save(write_file, data_dict)
    return None


def process_file(file):
    if file.startswith('f'):
        fid_func(file)
    elif file.startswith('O'):
        deriv_func(file)
    elif file.startswith('s'):
        deriv_func(file)
    return None


if __name__ == "__main__":   
    all_files = np.array(os.listdir(conv_dir))
    all_files = all_files[~np.isin(all_files, os.listdir(write_dir))]
    
    fid_inds = [i for i, s in enumerate(all_files) if s.startswith('f')]
    Om_inds = [i for i, s in enumerate(all_files) if s.startswith('O')]
    s8_inds = [i for i, s in enumerate(all_files) if s.startswith('s')]
    
    fid_files = all_files[fid_inds]
    Om_files = all_files[Om_inds]
    s8_files = all_files[s8_inds]
    
    # Remove any flipped or copy maps, not independent runs with the large mask surveys
    Om_crop_inds = [i for i, s in enumerate(Om_files) if (('flipped' not in s) & ('copy' not in s))]
    Om_files = Om_files[Om_crop_inds]
    
    test_files = np.append(np.append(fid_files, Om_files), s8_files)

    with Pool(50) as pool: # With 50 cores, used all 400 GB for 1:30
        res = pool.map(process_file, test_files)
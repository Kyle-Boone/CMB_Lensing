# This will be the code to generate the WST that will be used in the actual paper. I will not be using dyadic but instead will be rescaling. Due to the issues with noise in ACT, I will not be using the maps that Marco made and will instead add noise in my own scheme.

import sys
sys.path.append('/n/home09/kboone/software/Boone_Scattering/')

import os
import time
import numpy as np
import healpy as hp
import Boone_Scatter
from multiprocessing import Pool
from scipy.interpolate import interp1d


d_Om = 0.01
d_s8 = 0.015

nside = 1024
nside_ls = 3*nside

numS = 50
L = 4
M = 256
N = 256

write_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/wst_numS'+str(numS)+'_L'+str(L)+'/'

try:
    os.mkdir(write_dir)
except Exception:
    pass

conv_dir = '/n/netscratch/dvorkin_lab/Lab/gvalogiannis/maps_cmbl_georgios/maps_cmbl_georgios/'
mask_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/masks/'
patch_dir = '/n/home09/kboone/software/Data/Patches/'

survey = 'SPT'

mask_map = np.load(mask_dir + survey + '.npy')
    
SPT_surveys = ['SPT_main', 'SPT_summer']

SPT_mask_maps = []
for SPT_survey in SPT_surveys:
    SPT_mask_maps.append(np.load(mask_dir + SPT_survey + '.npy'))
    
SPT_weights = SPT_mask_maps[0]+SPT_mask_maps[1]

all_patch_inds = np.load(patch_dir + 'patches_nside_1024.npy')
weights = np.load(patch_dir + 'weights_' + survey + '.npy')
patch_inds = all_patch_inds[np.where(weights>0)]
unmask = patch_inds > -0.5


def antipode_map(m):
    nside = hp.get_nside(m)
    idx   = np.arange(m.size)
    x, y, z = hp.pix2vec(nside, idx)   # unit vectors
    idx2 = hp.vec2pix(nside, -x, -y, -z)
    return m[idx2]


def WST_map(survey_map):
    '''
    This function takes a healpix map (survey_map). It then calculates and returns a 
    data vector which contains WST coefficients up to order 2.
    '''
    st_calc = Boone_Scatter.Lin_Scattering2d(M=M, N=N, L=L, numS=numS)
    patches = np.zeros_like(patch_inds).astype(float)
    patches[unmask] = np.take(survey_map, patch_inds[unmask])
    
    imgs = patches
    for i in np.arange(len(imgs)):
        s_mean = st_calc.scattering_coef_simple(np.array([imgs[i]]))
        
        if i == 0:
            
            # This all assumes isotropy
            S1 = np.array(s_mean['S1_iso'])
            
            # Just isometry assumption
            S2 = np.atleast_2d(np.array(s_mean['S2_iso']).flatten())
            crop = ~np.isnan(S2[0])
            S2 = S2[:,crop]            
        else:
            
            # This all assumes isotropy
            S1_ind = np.array(s_mean['S1_iso'])
            
            # Just isotropy assumption
            S2_ind = np.atleast_2d(np.array(s_mean['S2_iso']).flatten())
            crop = ~np.isnan(S2_ind[0])
            S2_ind = S2_ind[:,crop]
            
            S1 = np.append(S1, S1_ind, axis=0)
            S2 = np.append(S2, S2_ind, axis=0)

    coeffs = np.append(S1, S2, axis=1)
    
    data_vec = np.sum(coeffs, axis = 0)
    return data_vec


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
    read_file = file.replace('_f', '')
    
    data = np.load(conv_dir+read_file, allow_pickle=True)
    map_p_unflipped = data[0]['full_sky_noiseless']
    map_m_unflipped = data[1]['full_sky_noiseless']
    
    map_p_noiseless = antipode_map(map_p_unflipped)
    map_m_noiseless = antipode_map(map_m_unflipped)
    
    data_dict = dict()
    
    nn = make_SPT_noise()
        
    cmb_map_p = map_p_noiseless + nn
    cmb_map_m = map_m_noiseless + nn
            
    cmb_map_p[np.where(mask_map < 0.5)] = 0
    cmb_map_m[np.where(mask_map < 0.5)] = 0
        
    wst_p = WST_map(cmb_map_p)
    wst_m = WST_map(cmb_map_m)
        
    data_dict[survey] = (wst_p - wst_m) / (2*d_param)
        
    np.save(write_file, data_dict)
    return None


if __name__ == "__main__":   
    all_files = np.array(os.listdir(conv_dir))
    
    Om_inds = [i for i, s in enumerate(all_files) if s.startswith('Om')]
    s8_inds = [i for i, s in enumerate(all_files) if s.startswith('s8')]
    
    Om_files = all_files[Om_inds]
    s8_files = all_files[s8_inds]
    # Remove any flipped or copy maps, not independent runs with the large mask surveys
    Om_crop_inds = [i for i, s in enumerate(Om_files) if (('flipped' not in s) & ('copy' not in s))]
    Om_files = Om_files[Om_crop_inds]
    
    Om_f_files = []
    s8_f_files = []
    for Om_file in Om_files:
        Om_num = file_num(Om_file)
        Om_f_files.append(Om_file.replace(Om_num, Om_num+'_f'))
    for s8_file in s8_files:
        s8_num = file_num(s8_file)
        s8_f_files.append(s8_file.replace(s8_num, s8_num+'_f'))
    Om_f_files = np.array(Om_f_files)
    s8_f_files = np.array(s8_f_files)
        
    Om_nums = []
    s8_nums = []
    for Om_file in Om_f_files:
        Om_nums.append(file_num(Om_file))
    for s8_file in s8_f_files:
        s8_nums.append(file_num(s8_file))
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
        Om_nums_written.append(file_num(Om_file))
    for s8_file in s8_files:
        s8_nums_written.append(file_num(s8_file))
    Om_nums_written = np.array(Om_nums_written)
    s8_nums_written = np.array(s8_nums_written)
    
    process_files = np.append(Om_f_files[~np.isin(Om_nums, Om_nums_written)], s8_f_files[~np.isin(s8_nums, s8_nums_written)])

    with Pool(17) as pool:
        res = pool.map(deriv_func, process_files)
import os
import numpy as np
import healpy as hp
import Boone_Scatter
from multiprocessing import Pool
from scipy.interpolate import interp1d

# WST Hyperparameters
# DO I WANT TO JUST CHANGE CODE SO THAT IT ONLY SAMPLES FROM L IN THE CASE OF S2 COEFFICIENTS IN WHICH CASE IT ONLY SAMPLES ONE? DOES THIS SAMPLING KILL SOME NON GAUSSIAN INFORMATION? It will kill correlations between different angles, not sure how important this is. Averaging over L isn't just an assumption of isotropy. For now, start with L=1 to get as many angles as possible.
numS = 7
L=1
M = 256
N = 256

write_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/wst_boone_numS'+str(numS)+'_l'+str(L)+'/'

try:
    os.mkdir(write_dir)
except Exception:
    pass

conv_dir = '/n/netscratch/dvorkin_lab/Lab/gvalogiannis/maps_cmbl_georgios/'
patch_dir = '/n/home09/kboone/software/Data/Patches/'

patch_inds = np.load(patch_dir + 'patches_nside_1024.npy')
coords = np.load(patch_dir + 'coords_nside_1024.npy')
unmask = patch_inds > -0.5

d_Om = 0.01
d_s8 = 0.015

nside = 1024

st_calc = Boone_Scatter.Lin_Scattering2d(M=M, N=N, L=L, numS=numS)


def make_SO_noise():
        # SO ----------------------------------------------------------------------------------------
        #  https://arxiv.org/abs/1808.07445
        N = 1e7
        x = [34.19233078337337	,71.86940802505276	,127.50974815233886	,208.08224631676606	,335.80966800506377	,626.9324561102427	,1193.7167598928752	,2274.5883052693794	,2734.7612461893104	,3075.370469000327	,4235.061816130342	]
        y = [0.1637325958704072,0.1658702835721177,0.19384826328334373,0.2923170686360547,0.38945080711492985,0.4148250745921913,0.5015602317089521,0.6742839831054106,1.1265018636427957,1.663103233757362,1.990063198637278]

        f = interp1d(x, y, kind='linear', fill_value="extrapolate")
        #f = scipy.interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
        cmb_lensing_noise_SO = hp.synfast(f(np.arange(3000))/N,nside =1024)
        return cmb_lensing_noise_SO


def WST_map(survey_map):
    '''
    This function takes a healpix map (survey_map). It then calculates and returns a 
    data vector which contains WST coefficients up to order 2.
    '''
    patches = np.zeros_like(patch_inds).astype(float)
    patches[unmask] = np.take(survey_map, patch_inds[unmask])
    
    imgs = patches
    for i in np.arange(len(imgs)):
        s_mean = st_calc.scattering_coef_simple(np.array([imgs[i]]))
        
        if i == 0:
            S0 = np.array(s_mean['S0'])
            # # Not assuming isotropy
            # S1 = np.atleast_2d(np.array(s_mean['S1']).flatten())
            # crop = ~np.isnan(S1[0])
            # S1 = S1[:,crop] 
            
            # S2 = np.atleast_2d(np.array(s_mean['S2']).flatten())
            # crop = ~np.isnan(S2[0])
            # S2 = S2[:,crop] 
            
            # This all assumes isotropy
            S1 = np.array(s_mean['S1_iso'])
            
            # Average over second L
            # S2 = np.average(np.array(s_mean['S2_iso']), axis = 3)
            # S2 = S2.reshape(1, J**2)
            # crop = ~np.isnan(S2[0])
            # S2 = S2[:,crop]
            
            # Just isometry assumption
            S2 = np.atleast_2d(np.array(s_mean['S2_iso']).flatten())
            crop = ~np.isnan(S2[0])
            S2 = S2[:,crop]            
        else:
            S0_ind = np.array(s_mean['S0'])
            
            # # Not assuming isotropy
            # S1_ind = np.atleast_2d(np.array(s_mean['S1']).flatten())
            # crop = ~np.isnan(S1_ind[0])
            # S1_ind = S1_ind[:,crop] 
            
            # S2_ind = np.atleast_2d(np.array(s_mean['S2']).flatten())
            # crop = ~np.isnan(S2_ind[0])
            # S2_ind = S2_ind[:,crop]
            
            # This all assumes isotropy
            S1_ind = np.array(s_mean['S1_iso'])
            
            # Average over second L
            # S2_ind = np.average(np.array(s_mean['S2_iso']), axis = 3)
            # S2_ind = S2_ind.reshape(1, J**2)
            # crop = ~np.isnan(S2_ind[0])
            # S2_ind = S2_ind[:,crop]
            
            # Just isometry assumption
            S2_ind = np.atleast_2d(np.array(s_mean['S2_iso']).flatten())
            crop = ~np.isnan(S2_ind[0])
            S2_ind = S2_ind[:,crop]
            
            S0 = np.append(S0, S0_ind, axis=0)
            S1 = np.append(S1, S1_ind, axis=0)
            S2 = np.append(S2, S2_ind, axis=0)

    coeffs = np.append(S0, np.append(S1, S2, axis=1), axis=1)
    
    data_vec = np.sum(coeffs, axis = 0)
    return data_vec


def file_num(s):
    # Find the index of the first '_'
    start = s.find('_')
    
    # Find the index of the first '.' after the '_'
    end = s.find('.', start)
    
    # Extract and return the substring between '_' and '.'
    return s[start + 1:end]


def fid_func(file):
    num = file_num(file)
    fid = np.load(conv_dir+file, allow_pickle=True).item()['full_sky_noiseless']
    nn = make_SO_noise()
    survey_map = fid+nn
    
    wst = WST_map(survey_map)
    np.save(write_dir + 'fiducial_'+num+'.npy', wst)
    return None


def Om_func(file):
    num = file_num(file)    
    data = np.load(conv_dir+file, allow_pickle=True)
    Om_plus = data[0]['full_sky_noiseless']
    Om_minus = data[1]['full_sky_noiseless']
    nn = make_SO_noise()
    survey_plus = Om_plus + nn
    survey_minus = Om_minus + nn
    
    wst_plus = WST_map(survey_plus)
    wst_minus = WST_map(survey_minus)
    
    wst_deriv = (wst_plus-wst_minus) / (2*d_Om)
    
    np.save(write_dir + 'Om_'+num+'.npy', wst_deriv)
    return None


def s8_func(file):
    num = file_num(file)    
    data = np.load(conv_dir+file, allow_pickle=True)
    s8_plus = data[0]['full_sky_noiseless']
    s8_minus = data[1]['full_sky_noiseless']
    nn = make_SO_noise()
    survey_plus = s8_plus + nn
    survey_minus = s8_minus + nn
    
    wst_plus = WST_map(survey_plus)
    wst_minus = WST_map(survey_minus)
    
    wst_deriv = (wst_plus-wst_minus) / (2*d_s8)
    
    np.save(write_dir + 's8_'+num+'.npy', wst_deriv)
    return None


def process_file(file):
    if file.startswith('f'):
        fid_func(file)
    elif file.startswith('O'):
        Om_func(file)
    elif file.startswith('s'):
        s8_func(file)
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
    test_files = np.append(np.append(fid_files, Om_files), s8_files)
    # test_files = np.append(Om_files, s8_files)
    
    with Pool(processes=32) as pool:
        res = pool.map(process_file, test_files)

# Changes nesessary for other surveys, try to make more general.
# No point in using Hartlap, that's just a constant scaling.
import sys
sys.path.append('/n/home09/kboone/CompressedFisher/')

from scipy.optimize import differential_evolution, minimize
from multiprocessing import Pool
from Bin_Helpers import *
import CompressedFisher
import numpy as np
import os


num_Derivs = 100
surveys = ['Planck_x_dr3']

for survey in surveys:
    per = 90

    wph_data_len = 392
    
    lmin = 2
    lmax = 3072 # Must be 3072 or less
    cro_data_len = (lmax-lmin) * 2 # Two tomographic bins
    
    numBins = 15
    compress_frac_split = 0.5
    realizations = 100000 # This is for realizations on the combined Fisher
    deriv_params = ['Om', 's8']

    sortInds = np.load('/n/home09/kboone/software/Data/MCMC_Sorting/WPH_J7_L4_Inds.npy')

    piv_dir = '/n/home09/kboone/software/Half_Data/Splits_Data/Cross/'
    write_file = piv_dir+'Both_'+str(num_Derivs)+'_'+survey+'_contours.npy'

    wph_piv_file = piv_dir+'WPH_'+str(num_Derivs)+'_'+survey+'.npy'
    all_wph_pivots = np.load(wph_piv_file)
    
    cro_piv_file = piv_dir+'Cl_'+str(num_Derivs)+'_'+survey+'.npy'
    all_cro_pivots = np.load(cro_piv_file)

    fid_split_file = piv_dir+'Fid_Splits.npy'
    all_fid_splits = np.load(fid_split_file)

    deriv_split_file = piv_dir+'Deriv_Splits_'+str(num_Derivs)+'.npy'
    all_deriv_splits = np.load(deriv_split_file)

    wph_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_x_wl/'+survey+'_wph_j7_l4/'

    wph_files = np.array(os.listdir(wph_dir))
    wph_fid_inds = [i for i, s in enumerate(wph_files) if s.startswith('f')]
    wph_fid_files = wph_files[wph_fid_inds]
    wph_Om_inds = [i for i, s in enumerate(wph_files) if s.startswith('O')]
    wph_s8_inds = [i for i, s in enumerate(wph_files) if s.startswith('s')]
    wph_Om_files = wph_files[wph_Om_inds]
    wph_s8_files = wph_files[wph_s8_inds]
    wph_Om_files = np.array([name for name in wph_Om_files if 'f' not in name])
    wph_s8_files = np.array([name for name in wph_s8_files if 'f' not in name])

    wph_fid = np.zeros((len(wph_fid_files), wph_data_len), dtype = np.float64)
    for i in np.arange(len(wph_fid)):
        wph_fid[i] = (np.load(wph_dir + wph_fid_files[i], allow_pickle=True)[:wph_data_len].real)[sortInds]

    wph_ave_fid = np.average(wph_fid, axis=0)
    wph_std_fid = np.std(wph_fid, axis=0)
    wph_fid = (wph_fid-wph_ave_fid) / wph_std_fid

    wph_partial_Om = np.zeros((num_Derivs, wph_data_len), dtype = np.float64)
    for i in np.arange(num_Derivs):
        wph_partial_Om[i] = (np.load(wph_dir + wph_Om_files[i], allow_pickle=True)[:wph_data_len].real)[sortInds]

    wph_partial_Om = wph_partial_Om / wph_std_fid

    wph_partial_s8 = np.zeros((num_Derivs, wph_data_len), dtype = np.float64)
    for i in np.arange(num_Derivs):
        wph_partial_s8[i] = (np.load(wph_dir + wph_s8_files[i], allow_pickle=True)[:wph_data_len].real)[sortInds]

    wph_partial_s8 = wph_partial_s8 / wph_std_fid
    
    
    cro_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_x_wl/raw_cross_cls/'

    cro_files = np.array(os.listdir(cro_dir))
    cro_fid_inds = [i for i, s in enumerate(cro_files) if s.startswith('f')]
    cro_fid_files = cro_files[cro_fid_inds]
    cro_Om_inds = [i for i, s in enumerate(cro_files) if s.startswith('O')]
    cro_s8_inds = [i for i, s in enumerate(cro_files) if s.startswith('s')]
    cro_Om_files = cro_files[cro_Om_inds]
    cro_s8_files = cro_files[cro_s8_inds]
    cro_Om_files = np.array([name for name in cro_Om_files if 'f' not in name])
    cro_s8_files = np.array([name for name in cro_s8_files if 'f' not in name])

    cro_fid = np.zeros((len(cro_fid_files), cro_data_len), dtype = np.float64)
    for i in np.arange(len(cro_fid)):
        cro_fid[i] = np.append(np.load(cro_dir + cro_fid_files[i], allow_pickle=True).item()[survey+'_low_z'][lmin:lmax], np.load(cro_dir + cro_fid_files[i], allow_pickle=True).item()[survey+'_high_z'][lmin:lmax])

    cro_ave_fid = np.average(cro_fid, axis=0)
    cro_std_fid = np.std(cro_fid, axis=0)
    cro_fid = (cro_fid-cro_ave_fid) / cro_std_fid

    cro_partial_Om = np.zeros((num_Derivs, cro_data_len), dtype = np.float64)
    for i in np.arange(num_Derivs):
        cro_partial_Om[i] = np.append(np.load(cro_dir + cro_Om_files[i], allow_pickle=True).item()[survey+'_low_z'][lmin:lmax], np.load(cro_dir + cro_Om_files[i], allow_pickle=True).item()[survey+'_high_z'][lmin:lmax])
    cro_partial_Om = cro_partial_Om / cro_std_fid

    cro_partial_s8 = np.zeros((num_Derivs, cro_data_len), dtype = np.float64)
    for i in np.arange(num_Derivs):
        cro_partial_s8[i] = np.append(np.load(cro_dir + cro_s8_files[i], allow_pickle=True).item()[survey+'_low_z'][lmin:lmax], np.load(cro_dir + cro_s8_files[i], allow_pickle=True).item()[survey+'_high_z'][lmin:lmax])
    cro_partial_s8 = cro_partial_s8 / cro_std_fid


    def get_contours(realization):
        fid_split = ~all_fid_splits[realization]
        deriv_split = ~all_deriv_splits[realization]
        wph_pivots = all_wph_pivots[realization]
        cro_pivots = all_cro_pivots[realization]

        wph_Q = create_linlin_matrix(wph_data_len, numBins, points=wph_pivots)
        cro_Q = create_linlin_matrix(cro_data_len, numBins, points=cro_pivots)

        fid = np.append(wph_fid[fid_split] @ wph_Q, cro_fid[fid_split] @ cro_Q, axis=1)
        deriv_dict = dict()
        deriv_dict['Om'] = np.append(wph_partial_Om[deriv_split] @ wph_Q, cro_partial_Om[deriv_split] @ cro_Q, axis=1)
        deriv_dict['s8'] = np.append(wph_partial_s8[deriv_split] @ wph_Q, cro_partial_s8[deriv_split] @ cro_Q, axis=1)

        cov = get_comb_cov(fid, deriv_dict, compress_frac_split, realizations=realizations, median=False, perUse=(per/100))
        return cov


    nums = np.arange(len(all_deriv_splits))
    with Pool(50) as pool:
        res = pool.map(get_contours, nums)
    all_pivots = np.array(res)
    np.save(write_file, all_pivots)
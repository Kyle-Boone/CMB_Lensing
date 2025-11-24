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


num_Derivs = 25
surveys = ['ACT', 'Planck', 'SO', 'SPT']

for survey in surveys:
    dim = 6
    per = 90

    data_len = 5150
    numBins = 15
    compress_frac_split = 0.5
    realizations = 100000 # This is for realizations on the combined Fisher
    deriv_params = ['Om', 's8']
    
    sortInds = np.load('/n/home09/kboone/software/Data/MCMC_Sorting/WST_S50_L4_Inds.npy')

    piv_dir = '/n/home09/kboone/software/Half_Data/Splits_Data/CMBL/'
    write_file = piv_dir+'WST_'+str(num_Derivs)+'_'+survey+'_contours.npy'

    piv_file = piv_dir+'WST_'+str(num_Derivs)+'_'+survey+'.npy'
    all_pivots = np.load(piv_file)

    fid_split_file = piv_dir+'Fid_Splits.npy'
    all_fid_splits = np.load(fid_split_file)

    deriv_split_file = piv_dir+'Deriv_Splits_'+str(num_Derivs)+'.npy'
    all_deriv_splits = np.load(deriv_split_file)

    fid_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/wst_numS50_L4/'
    deriv_dir = fid_dir

    all_files = np.array(os.listdir(fid_dir))
    fid_inds = [i for i, s in enumerate(all_files) if s.startswith('f')]
    all_fid_files = all_files[fid_inds]

    all_files = np.array(os.listdir(deriv_dir))
    Om_inds = [i for i, s in enumerate(all_files) if s.startswith('O')]
    s8_inds = [i for i, s in enumerate(all_files) if s.startswith('s')]
    all_Om_files = all_files[Om_inds]
    all_s8_files = all_files[s8_inds]
    all_Om_files = np.array([name for name in all_Om_files if 'f' not in name])
    all_s8_files = np.array([name for name in all_s8_files if 'f' not in name])

    all_fid = np.zeros((len(all_fid_files), data_len), dtype = np.float64)
    for i in np.arange(len(all_fid)):
        all_fid[i] = (np.load(fid_dir + all_fid_files[i], allow_pickle=True).item()[survey][:data_len])[sortInds]

    ave_fid = np.average(all_fid, axis=0)
    std_fid = np.std(all_fid, axis=0)
    all_fid = (all_fid-ave_fid) / std_fid

    all_partial_Om = np.zeros((num_Derivs, data_len), dtype = np.float64)
    for i in np.arange(num_Derivs):
        all_partial_Om[i] = (np.load(deriv_dir + all_Om_files[i], allow_pickle=True).item()[survey][:data_len])[sortInds]

    all_partial_Om = all_partial_Om / std_fid

    all_partial_s8 = np.zeros((num_Derivs, data_len), dtype = np.float64)
    for i in np.arange(num_Derivs):
        all_partial_s8[i] = (np.load(deriv_dir + all_s8_files[i], allow_pickle=True).item()[survey][:data_len])[sortInds]

    all_partial_s8 = all_partial_s8 / std_fid


    def get_contours(realization):
        fid_split = ~all_fid_splits[realization]
        deriv_split = ~all_deriv_splits[realization]
        pivots = all_pivots[realization]

        Q = create_linlin_matrix(data_len, numBins, points=pivots)
        Q_orig = create_linlin_matrix(data_len, numBins)

        fid = all_fid[fid_split] @ Q
        deriv_dict = dict()
        deriv_dict['Om'] = all_partial_Om[deriv_split] @ Q
        deriv_dict['s8'] = all_partial_s8[deriv_split] @ Q

        fid_orig = all_fid[fid_split] @ Q_orig
        deriv_dict_orig = dict()
        deriv_dict_orig['Om'] = all_partial_Om[deriv_split] @ Q_orig
        deriv_dict_orig['s8'] = all_partial_s8[deriv_split] @ Q_orig

        cov = get_comb_cov(fid, deriv_dict, compress_frac_split, realizations=realizations, median=False, perUse=(per/100))
        return cov # Use this when not using all derivatives.
        # cov_orig = get_comb_cov(fid_orig, deriv_dict_orig, compress_frac_split, realizations=realizations, median=False, perUse=(per/100))
        # return cov_orig


    nums = np.arange(1000)
    with Pool(50) as pool:
        res = pool.map(get_contours, nums)
    all_pivots = np.array(res)
    np.save(write_file, all_pivots)
# Changes nesessary for other surveys, try to make more general.
# No point in using Hartlap, that's just a constant scaling.
from scipy.optimize import differential_evolution, minimize
from multiprocessing import Pool
from Bin_Helpers import *
import numpy as np
import os


num_Derivs = 25
surveys = ['ACT', 'Planck', 'SO', 'SPT']

for survey in surveys:

    nside = 1024
    lmin = 2
    lmax = 3072 # Must be 3072 or less
    data_len = lmax-lmin
    numBins = 15

    piv_dir = '/n/home09/kboone/software/Half_Data/Splits_Data/CMBL/'
    try:
        os.mkdir(piv_dir)
    except Exception:
        pass

    len_x = 1
    len_y = 1

    dim = len_x + len_y

    write_file = piv_dir+'Cl_'+str(num_Derivs)+'_'+survey+'.npy'

    all_fid_splits = np.load(piv_dir+'Fid_Splits.npy')
    all_deriv_splits = np.load(piv_dir+'Deriv_Splits_'+str(num_Derivs)+'.npy')

    fid_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_conv/cls/'
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
        all_fid[i] = np.load(fid_dir + all_fid_files[i], allow_pickle=True).item()[survey][lmin:lmax]

    ave_fid = np.average(all_fid, axis=0)
    std_fid = np.std(all_fid, axis=0)
    all_fid = (all_fid-ave_fid) / std_fid

    all_partial_Om = np.zeros((num_Derivs, data_len), dtype = np.float64)
    for i in np.arange(num_Derivs):
        all_partial_Om[i] = np.load(deriv_dir + all_Om_files[i], allow_pickle=True).item()[survey][lmin:lmax]

    all_partial_Om = all_partial_Om / std_fid

    all_partial_s8 = np.zeros((num_Derivs, data_len), dtype = np.float64)
    for i in np.arange(num_Derivs):
        all_partial_s8[i] = np.load(deriv_dir + all_s8_files[i], allow_pickle=True).item()[survey][lmin:lmax]

    all_partial_s8 = all_partial_s8 / std_fid

    XL, XH = 0, data_len-1               # x-range of the tophat box
    YL, YH = 0, numBins-1                # y–range of the tophat box


    def find_pivots(realization):
        fid_split = all_fid_splits[realization]
        deriv_split = all_deriv_splits[realization]

        fid = all_fid[fid_split]
        partial_Om = all_partial_Om[deriv_split]
        partial_s8 = all_partial_s8[deriv_split]
        partial_Om = np.atleast_2d(np.average(partial_Om, axis = 0))
        partial_s8 = np.atleast_2d(np.average(partial_s8, axis = 0))

        C = np.atleast_2d(np.cov(fid.T, bias=True))
        D = np.append(partial_Om, partial_s8, axis = 0).T


        def f(x_pivots, y_pivots):
            y_pivots_sorted = np.sort(y_pivots)
            x_pivots_sorted = np.sort(x_pivots)
            for y_pivot in y_pivots_sorted:
                if (y_pivot > YH) | (y_pivot < YL):
                    return 0
            # Checking for issues with invertibility
            for i in np.arange(len(y_pivots) - 1):
                if y_pivots_sorted[i+1] - y_pivots_sorted[i] > x_pivots_sorted[i+1] - x_pivots_sorted[i]:
                    return 0
            if y_pivots_sorted[0] > x_pivots_sorted[0]:
                return 0
            if YH - y_pivots_sorted[-1] > XH - x_pivots_sorted[-1]:
                return 0
            pivots = (np.append(np.atleast_2d(x_pivots_sorted), np.atleast_2d(y_pivots_sorted), axis=0).T).tolist()
            return -np.log(det_fish(create_linlin_matrix(data_len, numBins, points=pivots), C, D)) # This is what I want to maximize.


        def eval_f(all_pivots):
            x_pivots = all_pivots[:len_x]
            y_pivots = all_pivots[len_x:]
            return f(x_pivots, y_pivots)


        def optimize(popsize=20, max_de_iter=10000, max_nm_iter=2000):
            x_bounds = [(0, XH)] * len_x
            y_bounds = [(0, YH)] * len_y
            bounds = x_bounds + y_bounds

            # Global exploration with Differential Evolution
            de_result = differential_evolution(
                eval_f,  # maximize f by minimizing -f
                bounds,
                strategy='best1bin',
                popsize=popsize,
                maxiter=max_de_iter,
                atol=1e-1,
                polish=False,
                workers=1
            )

            # Local Nelder-Mead refinement
            nm_result = minimize(
                eval_f,
                de_result.x,
                method='Nelder-Mead',
                options={'maxiter': max_nm_iter, 'xatol': 1e-2, 'disp': True}
            )

            final_params = nm_result.x
            final_val = eval_f(final_params)
            return final_params, final_val

        best_params, best_value = optimize()

        x_params = best_params[:len_x]
        y_params = best_params[len_x:]

        x_params = np.sort(x_params)
        y_params = np.sort(y_params)

        best_pivots = np.append(np.atleast_2d(x_params), np.atleast_2d(y_params), axis=0).T
        return best_pivots


    nums = np.arange(1000)
    with Pool(50) as pool:
        res = pool.map(find_pivots, nums)
    all_pivots = np.array(res)
    np.save(write_file, all_pivots)
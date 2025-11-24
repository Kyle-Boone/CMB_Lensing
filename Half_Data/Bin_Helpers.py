import sys
sys.path.append('/n/home09/kboone/CompressedFisher/')

import os
import numpy as np
import CompressedFisher
from getdist import MCSamples
from scipy.ndimage import gaussian_filter


def get_conv(fid, deriv_dict, compress_frac_split, fracs=None, mode = 'Combined'):
    # Mode should be 'Combined', 'Compressed', or 'Standard'
    deriv_params = list(deriv_dict.keys())
    nSims_deriv = len(deriv_dict[deriv_params[0]][:,0])
    cFisher = CompressedFisher.gaussianFisher(deriv_params,nSims_deriv)
    cFisher.initailize_covmat(fid)
    cFisher.initailize_mean(fid)
    cFisher.initailize_deriv_sims(dic_deriv_sims=deriv_dict)
    if mode == 'Combined':
        return cFisher.run_combined_fisher_deriv_stablity_test(deriv_params,compress_frac_split, sample_fractions=fracs, max_repeats=1)
    elif mode == 'Compressed':
        return cFisher.run_compressed_fisher_deriv_stablity_test(deriv_params,compress_frac_split, sample_fractions=fracs, max_repeats=1)
    elif mode == 'Standard':
        return cFisher.run_fisher_deriv_stablity_test(deriv_params, sample_fractions=fracs, max_repeats=1)
    else:
        print('Invalid mode')
        return None


def get_comb_cov(fid, deriv_dict, compress_frac_split, mode = 'Combined', realizations=1, median=True, perUse = 1):
    # Mode should be 'Combined', 'Compressed', or 'Standard'
    deriv_params = list(deriv_dict.keys())
    nSims_deriv = len(deriv_dict[deriv_params[0]][:,0])
    covs = np.zeros((realizations, len(deriv_dict), len(deriv_dict)), dtype=np.float64)
    for i in np.arange(realizations):
        cFisher = CompressedFisher.gaussianFisher(deriv_params,nSims_deriv)
        cFisher.initailize_covmat(fid)
        cFisher.initailize_mean(fid)
        cFisher.initailize_deriv_sims(dic_deriv_sims=deriv_dict)
        cFisher.generate_deriv_sim_splits(compress_frac_split)
        if mode == 'Combined':
            covs[i] = cFisher.compute_combined_fisher_forecast(deriv_params)
        elif mode == 'Compressed':
            covs[i] = cFisher.compute_compressed_fisher_forecast(deriv_params)
        elif mode == 'Standard':
            covs[i] = cFisher.compute_fisher_forecast(deriv_params)
        else:
            print('Invalid mode')
            return None
    if perUse >= 1:
        if median:
            return np.median(covs, axis=0)
        else:
            return np.average(covs, axis=0)
    else:
        crop = (1-perUse)/2
        covs = covs[np.argsort(np.linalg.det(covs))]
        covs = covs[int(crop*len(covs)):int((1-crop)*len(covs))]
        if median:
            return np.median(covs, axis=0)
        else:
            return np.average(covs, axis=0)


def get_wst_data(fid_dir, deriv_dir, deriv_params, data_len, compression_matrix, survey, sortInds, frac_fid=1., frac_deriv=1., print_derivs=False, flip=False):
    
    numBins = len(compression_matrix[:,0])
    
    all_files = np.array(os.listdir(fid_dir))
    fid_inds = [i for i, s in enumerate(all_files) if s.startswith('f')]
    fid_files = all_files[fid_inds]
    if frac_fid<1.:
        np.random.shuffle(fid_files)
        fid_files=fid_files[:int(frac_fid * len(fid_files))]
    
    all_files = np.array(os.listdir(deriv_dir))
    deriv_files = []
    for deriv_param in deriv_params:
        deriv_inds = [i for i, s in enumerate(all_files) if s.startswith(deriv_param[0])]
        deriv_files_ind = all_files[deriv_inds]
        if not flip:
            deriv_files_ind = [name for name in deriv_files_ind if 'f' not in name]
        if frac_deriv<1:
            np.random.shuffle(deriv_files_ind)
            deriv_files_ind=deriv_files_ind[:int(frac_deriv * len(deriv_files_ind))]
        deriv_files.append(deriv_files_ind)
    if print_derivs:
        print(len(deriv_files[0]))
        
    fid = np.zeros((len(fid_files), numBins), dtype = np.float64)
    for i in np.arange(len(fid_files)):
        data = (np.load(fid_dir + fid_files[i], allow_pickle=True).item()[survey][:data_len])[sortInds]
        fid[i] = compression_matrix @ data
    ave_fid = np.average(fid, axis=0)
    std_fid = np.std(fid, axis=0)
    fid = (fid-ave_fid) / std_fid
        
    deriv_dict = {}
    for i in np.arange(len(deriv_params)):
        deriv = np.zeros((len(deriv_files[i]), numBins), dtype = np.float64)
        for j in np.arange(len(deriv_files[i])):
            data = (np.load(deriv_dir + deriv_files[i][j], allow_pickle=True).item()[survey][:data_len])[sortInds]
            data = compression_matrix @ data
            data = data / std_fid
            deriv[j] = data
        deriv_dict[deriv_params[i]] = deriv    
        
    return fid, deriv_dict
    
    
def get_cl_data(fid_dir, deriv_dir, deriv_params, data_len, compression_matrix, survey, frac_fid=1., frac_deriv=1., print_derivs=False, flip=False):
    
    numBins = len(compression_matrix[:,0])
    
    all_files = np.array(os.listdir(fid_dir))
    fid_inds = [i for i, s in enumerate(all_files) if s.startswith('f')]
    fid_files = all_files[fid_inds]
    if frac_fid<1.:
        np.random.shuffle(fid_files)
        fid_files=fid_files[:int(frac_fid * len(fid_files))]
    
    all_files = np.array(os.listdir(deriv_dir))
    deriv_files = []
    for deriv_param in deriv_params:
        deriv_inds = [i for i, s in enumerate(all_files) if s.startswith(deriv_param[0])]
        deriv_files_ind = all_files[deriv_inds]
        if not flip:
            deriv_files_ind = [name for name in deriv_files_ind if 'f' not in name]
        if frac_deriv<1:
            np.random.shuffle(deriv_files_ind)
            deriv_files_ind=deriv_files_ind[:int(frac_deriv * len(deriv_files_ind))]
        deriv_files.append(deriv_files_ind)
    if print_derivs:
        print(len(deriv_files[0]))
        
    fid = np.zeros((len(fid_files), numBins), dtype = np.float64)
    for i in np.arange(len(fid_files)):
        data = np.load(fid_dir + fid_files[i], allow_pickle=True).item()[survey][:data_len]
        fid[i] = compression_matrix @ data
    ave_fid = np.average(fid, axis=0)
    std_fid = np.std(fid, axis=0)
    fid = (fid-ave_fid) / std_fid
        
    deriv_dict = {}
    for i in np.arange(len(deriv_params)):
        deriv = np.zeros((len(deriv_files[i]), numBins), dtype = np.float64)
        for j in np.arange(len(deriv_files[i])):
            data = np.load(deriv_dir + deriv_files[i][j], allow_pickle=True).item()[survey][:data_len]
            data = compression_matrix @ data
            data = data / std_fid
            deriv[j] = data
        deriv_dict[deriv_params[i]] = deriv    
        
    return fid, deriv_dict


def get_data(fid_dir, deriv_dir, deriv_params, data_len, compression_matrix, survey=None, sortInds=None, cross_cl=False, frac_fid=1., frac_deriv=1., flip=False, print_derivs=True):
    if cross_cl:
        if survey is None:
            print('If doing cross cl, a survey name is necessary')
            return None
        
    numBins = len(compression_matrix[:,0])
        
    all_files = np.array(os.listdir(fid_dir))
    fid_inds = [i for i, s in enumerate(all_files) if s.startswith('f')]
    fid_files = all_files[fid_inds]
    if frac_fid<1.:
        np.random.shuffle(fid_files)
        fid_files=fid_files[:int(frac_fid * len(fid_files))]
    
    all_files = np.array(os.listdir(deriv_dir))
    deriv_files = []
    for deriv_param in deriv_params:
        deriv_inds = [i for i, s in enumerate(all_files) if s.startswith(deriv_param[0])]
        deriv_files_ind = all_files[deriv_inds]
        if not flip:
            deriv_files_ind = [name for name in deriv_files_ind if 'f' not in name]
        if frac_deriv<1:
            np.random.shuffle(deriv_files_ind)
            deriv_files_ind=deriv_files_ind[:int(frac_deriv * len(deriv_files_ind))]
        deriv_files.append(deriv_files_ind)
    if print_derivs:
        print(len(deriv_files[0]))
        
    fid = np.zeros((len(fid_files), numBins), dtype = np.float64)
    for i in np.arange(len(fid_files)):
        if survey is None:
            data = np.load(fid_dir + fid_files[i], allow_pickle=True)[:data_len].real
        else:
            if cross_cl:
                data = np.append(np.load(fid_dir + fid_files[i], allow_pickle=True).item()[survey+'_low_z'][:int(data_len/2)], np.load(fid_dir + fid_files[i], allow_pickle=True).item()[survey+'_high_z'][:int(data_len/2)])
            else:
                data = np.load(fid_dir + fid_files[i], allow_pickle=True).item()[survey][:data_len]
        if sortInds is not None:
            data = data[sortInds]
        fid[i] = compression_matrix @ data
    ave_fid = np.average(fid, axis=0)
    std_fid = np.std(fid, axis=0)
    fid = (fid-ave_fid) / std_fid
    
    deriv_dict = {}
    for i in np.arange(len(deriv_params)):
        deriv = np.zeros((len(deriv_files[i]), numBins), dtype = np.float64)
        for j in np.arange(len(deriv_files[i])):
            if survey is None:
                data = np.load(deriv_dir + deriv_files[i][j], allow_pickle=True)[:data_len].real
            else:
                if cross_cl:
                    data = np.append(np.load(deriv_dir + deriv_files[i][j], allow_pickle=True).item()[survey+'_low_z'][:int(data_len/2)], np.load(deriv_dir + deriv_files[i][j], allow_pickle=True).item()[survey+'_high_z'][:int(data_len/2)])
                else:
                    data = np.load(deriv_dir + deriv_files[i][j], allow_pickle=True).item()[survey][:data_len]
            if sortInds is not None:
                data = data[sortInds]
            data = compression_matrix @ data
            data = data / std_fid
            deriv[j] = data
        deriv_dict[deriv_params[i]] = deriv    
        
    return fid, deriv_dict


def get_C_D(fid_dir, deriv_dir, data_len, numBins, surveys=None, sortInds=None, cross_cl=False, frac_fid=1., frac_Om=1., frac_s8=1.):
    
    if cross_cl:
        if surveys is None:
            print('If doing cross cl, a survey name is necessary')
            return None
        
    all_files = np.array(os.listdir(fid_dir))
    fid_inds = [i for i, s in enumerate(all_files) if s.startswith('f')]
    fid_files = all_files[fid_inds]
    if frac_fid<1.:
        np.random.shuffle(fid_files)
        fid_files=fid_files[:int(frac_fid * len(fid_files))]
    
    all_files = np.array(os.listdir(deriv_dir))
    Om_inds = [i for i, s in enumerate(all_files) if s.startswith('O')]
    s8_inds = [i for i, s in enumerate(all_files) if s.startswith('s')]
    Om_files = all_files[Om_inds]
    if frac_Om<1.:
        np.random.shuffle(Om_files)
        Om_files=Om_files[:int(frac_Om * len(Om_files))]
    s8_files = all_files[s8_inds]
    if frac_s8<1.:
        np.random.shuffle(s8_files)
        s8_files=s8_files[:int(frac_s8 * len(s8_files))]
    
    Cs = []
    Ds = []
    if surveys is None:
        Cls = np.zeros((len(fid_files), data_len), dtype = np.float64)
        for i in np.arange(len(fid_files)):
            data = np.load(fid_dir + fid_files[i], allow_pickle=True)[:data_len].real
            if sortInds is not None:
                data = data[sortInds]
            Cls[i] = data
        ave_Cls = np.average(Cls, axis=0)

        N = len(Cls)
        m = numBins

        C = np.atleast_2d(np.cov(Cls.T, bias=True)) * (N-1) / (N-m-2)
        max_C = np.max(C)
        C /= max_C
        
        partial_Om = np.zeros((len(Om_files), data_len), dtype = np.float64)
        for i in np.arange(len(Om_files)):
            data = np.load(deriv_dir + Om_files[i], allow_pickle=True)[:data_len].real
            if sortInds is not None:
                data = data[sortInds]
            partial_Om[i] = data
        partial_Om = np.atleast_2d(np.average(partial_Om, axis = 0))
        partial_Om = partial_Om#  / stds

        partial_s8 = np.zeros((len(s8_files), data_len), dtype = np.float64)
        for i in np.arange(len(s8_files)):
            data = np.load(deriv_dir + s8_files[i], allow_pickle=True)[:data_len].real
            if sortInds is not None:
                data = data[sortInds]
            partial_s8[i] = data
        partial_s8 = np.atleast_2d(np.average(partial_s8, axis = 0))
        partial_s8 = partial_s8#  / stds

        D = np.append(partial_Om, partial_s8, axis = 0).T
        D /= np.sqrt(max_C)
        
        Cs.append(C)
        Ds.append(D)
    else:
        for survey in surveys:
            Cls = np.zeros((len(fid_files), data_len), dtype = np.float64)
            for i in np.arange(len(fid_files)):
                if cross_cl:
                    data = np.append(np.load(fid_dir + fid_files[i], allow_pickle=True).item()[survey+'_low_z'][:int(data_len/2)], np.load(fid_dir + fid_files[i], allow_pickle=True).item()[survey+'_high_z'][:int(data_len/2)])
                else:
                    data = np.load(fid_dir + fid_files[i], allow_pickle=True).item()[survey][:data_len]
                if sortInds is not None:
                    data = data[sortInds]
                Cls[i] = data
            ave_Cls = np.average(Cls, axis=0)

            N = len(Cls)
            m = numBins

            C = np.atleast_2d(np.cov(Cls.T, bias=True)) * (N-1) / (N-m-2)
            max_C = np.max(C)
            C /= max_C

            partial_Om = np.zeros((len(Om_files), data_len), dtype = np.float64)
            for i in np.arange(len(Om_files)):
                if cross_cl:
                    data = np.append(np.load(deriv_dir + Om_files[i], allow_pickle=True).item()[survey+'_low_z'][:int(data_len/2)], np.load(deriv_dir + Om_files[i], allow_pickle=True).item()[survey+'_high_z'][:int(data_len/2)])
                else:
                    data = np.load(deriv_dir + Om_files[i], allow_pickle=True).item()[survey][:data_len]
                if sortInds is not None:
                    data = data[sortInds]
                partial_Om[i] = data
            partial_Om = np.atleast_2d(np.average(partial_Om, axis = 0))
            partial_Om = partial_Om#  / stds

            partial_s8 = np.zeros((len(s8_files), data_len), dtype = np.float64)
            for i in np.arange(len(s8_files)):
                if cross_cl:
                    data = np.append(np.load(deriv_dir + s8_files[i], allow_pickle=True).item()[survey+'_low_z'][:int(data_len/2)], np.load(deriv_dir + s8_files[i], allow_pickle=True).item()[survey+'_high_z'][:int(data_len/2)])
                else:
                    data = np.load(deriv_dir + s8_files[i], allow_pickle=True).item()[survey][:data_len]
                if sortInds is not None:
                    data = data[sortInds]
                partial_s8[i] = data
            partial_s8 = np.atleast_2d(np.average(partial_s8, axis = 0))
            partial_s8 = partial_s8#  / stds

            D = np.append(partial_Om, partial_s8, axis = 0).T
            D /= np.sqrt(max_C)

            Cs.append(C)
            Ds.append(D)
    return Cs, Ds


def get_chain(cov_mat, Om=0.3175, s8=0.834):
    chain = []
    #imagine you have multiple cases (different DV with different choices of scales)
    mean = [Om, s8]
    C_par = cov_mat

    x = np.random.multivariate_normal(np.array(mean),C_par,30000)
    sig8_ = np.array(x[:,1]).astype(np.float64)
    om_ = np.array(x[:,0]).astype(np.float64)       


    ssa = np.c_[om_.T,sig8_.T]
    samples_ = MCSamples(samples=ssa,weights=np.ones(30000), names = ['Om','sigma8'], labels = [r'\Omega_{\rm m}','\sigma_8'])

    chain.append(samples_)
    
    return chain


def create_linlog_matrix(params, data_len, numBins):
    new_bin, old_bin = params
    
    x = np.arange(data_len).astype(int)
    y = __linlog_binning__(x, new_bin, old_bin, data_len, numBins)

    return __create_matrix__(x, y, data_len, numBins)


def create_linlin_matrix(data_len, numBins, points=None):
    x = np.arange(data_len).astype(int)
    y = __gen_linlin__(x, points, data_len, numBins)
    
    return __create_matrix__(x, y, data_len, numBins)


def det_fish(mat, C, D):
    return np.linalg.det(D.T @ mat @ np.linalg.inv(mat.T @ C @ mat) @ mat.T @ D)


def sig_Om(mat, C, D):
    return np.sqrt(np.linalg.inv(D.T @ mat @ np.linalg.inv(mat.T @ C @ mat) @ mat.T @ D)[0,0])


def sig_s8(mat, C, D):
    return np.sqrt(np.linalg.inv(D.T @ mat @ np.linalg.inv(mat.T @ C @ mat) @ mat.T @ D)[1,1])


def fish(mat, C, D):
    return D.T @ mat @ np.linalg.inv(mat.T @ C @ mat) @ mat.T @ D


def gaussian_smooth_on_grid(x_grid, y_grid, f_grid, sigma_x, sigma_y,
                            indexing='ij', mode='constant'):
    """
    Smooth a 2D grid f_grid over coordinates (x_grid, y_grid) with Gaussian
    standard deviations sigma_x and sigma_y (in the same units as x_grid/y_grid).

    Parameters
    ----------
    x_grid, y_grid : array-like, shape (Nx, Ny)
        Meshgrid arrays produced with either:
          • indexing='ij': X[i,j]=x[i], Y[i,j]=y[j]
          • indexing='xy': X[i,j]=x[j], Y[i,j]=y[i]
    f_grid : 2D array, shape (Nx, Ny)
        Values defined on that grid.
    sigma_x, sigma_y : float
        Gaussian sigma in the x- and y-directions (physical units).
    indexing : {'ij', 'xy'}
        How your meshgrid was created.  Default is 'ij'.
    mode : str, optional
        Boundary mode for scipy.ndimage.gaussian_filter.

    Returns
    -------
    f_smooth : 2D array, shape (Nx, Ny)
        Smoothed grid.
    """
    # Extract the coordinate vectors
    if x_grid.ndim == 2:
        if indexing == 'ij':
            x_vals = x_grid[:, 0]
            y_vals = y_grid[0, :]
        else:  # 'xy'
            x_vals = x_grid[0, :]
            y_vals = y_grid[:, 0]
    else:
        x_vals = np.asarray(x_grid)
        y_vals = np.asarray(y_grid)

    # Compute average spacing
    dx = np.mean(np.diff(x_vals))
    dy = np.mean(np.diff(y_vals))

    # Convert to pixel units
    sigma_pix_x = sigma_x / dx
    sigma_pix_y = sigma_y / dy

    # scipy expects sigma per axis in array-order: (axis0, axis1)
    # For 'ij', axis0 corresponds to x and axis1 to y.
    # For 'xy', axis0→y, axis1→x, so we'd swap them internally.
    if indexing == 'ij':
        sigmas = (sigma_pix_x, sigma_pix_y)
    else:  # 'xy'
        sigmas = (sigma_pix_y, sigma_pix_x)

    # Apply the filter
    f_smooth = gaussian_filter(f_grid, sigma=sigmas, mode=mode)
    return f_smooth


def __linlog_binning__(x, new_bin, old_bin, data_len, numBins):
    # Linear until (old_bin, new_bin), log binning after
    x = np.array([x]).flatten()
    # Compute the exponential rate B:
    B = np.log((data_len-1) / old_bin) / (numBins - new_bin - 1) # Maps between 0 and 99, what we want
    
    # Inverse in the linear region: Y = (y/x)*t  => t = (x/y)*Y.
    # In the exponential region: Y = y * exp(B*(t-x))  => t = x + (1/B)*ln(Y/y).
    retvals = np.zeros(len(x)).astype(float)
    retvals[np.where(x<=old_bin)[0]] = (new_bin / old_bin) * x[np.where(x<=old_bin)[0]]
    retvals[np.where(x>old_bin)[0]] = new_bin + np.log(x[np.where(x>old_bin)[0]] / old_bin) / B
    return retvals


def __gen_linlin__(x, points, data_len, numBins):
    """
    Interpolates through all points
    """
    # Sort the points by their x-coordinate
    if points is None:
        xp = [0, data_len-1]
        yp = [0, numBins-1]
    else:
        sorted_points = sorted(points, key=lambda pt: pt[0])

        # Create lists for x and y values including the endpoints
        xp = [0] + [pt[0] for pt in sorted_points] + [data_len-1]
        yp = [0] + [pt[1] for pt in sorted_points] + [numBins-1]
    
    # Use numpy.interp to compute the interpolated values
    y = np.interp(x, xp, yp)
    return y


def __create_matrix__(x, y, data_len, numBins):
    M = np.zeros((data_len, numBins), dtype=float)
    
    # Compute the floor indices (left bin)
    left_bins = np.floor(y).astype(int)
    # Compute the next bin index (right bin)
    right_bins = left_bins + 1
    # Compute the fractional part
    frac = y - left_bins

    maskRight = right_bins >= numBins
    maskLeft = left_bins < 0
    
    maskNeither = (~maskRight)&(~maskLeft)
    maskOnlyLeft = (~maskRight)&(maskLeft)
    maskOnlyRight = (maskRight)&(~maskLeft)
    
    # Neither side is masked, both within bounds
    M[x[maskNeither], left_bins[maskNeither]] = 1.0 - frac[maskNeither]
    M[x[maskNeither], right_bins[maskNeither]] = frac[maskNeither]
    
    # For points where left_bins are within bounds only
    M[x[maskOnlyRight], left_bins[maskOnlyRight]] = 1.0 - frac[maskOnlyRight]
    
    # For points where right_bins are within bounds only
    M[x[maskOnlyLeft], right_bins[maskOnlyLeft]] = frac[maskOnlyLeft]

    return M
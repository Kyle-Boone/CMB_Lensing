import multiprocessing as mp
import numpy as np
import emcee
import h5py
import os


Om = 0.3175

s8 = 0.834

lmax = 2000
numBins = 100

survey = 'Planck_x_dr3'

write_file = '/n/home09/kboone/software/Data/MCMC_Bins/'+survey+'_Dim5.h5'


def gen_linlin(x, points, lmax, numBins):
    """
    Interpolates through all points
    """
    # Sort the points by their x-coordinate
    sorted_points = sorted(points, key=lambda pt: pt[0])
    
    # Create lists for x and y values including the endpoints
    xp = [0] + [pt[0] for pt in sorted_points] + [lmax]
    yp = [0] + [pt[1] for pt in sorted_points] + [numBins-1]
    
    # Use numpy.interp to compute the interpolated values
    y = np.interp(x, xp, yp)
    return y


def create_matrix(x, y, lmax, numBins):
    M = np.zeros((lmax+1, numBins), dtype=float)
    
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


def create_linlin_matrix(points, lmax, numBins=numBins):
    x = np.arange(lmax+1).astype(int)
    y = gen_linlin(x, points, lmax, numBins)
    
    return create_matrix(x, y, lmax, numBins)


fid_dir = '/n/netscratch/dvorkin_lab/Lab/kboone/cmb_x_wl/raw_cross_cls/'
all_files = np.array(os.listdir(fid_dir))
fid_inds = [i for i, s in enumerate(all_files) if s.startswith('f')]
fid_files = all_files[fid_inds]

deriv_dir = fid_dir
all_files = np.array(os.listdir(deriv_dir))
Om_inds = [i for i, s in enumerate(all_files) if s.startswith('O')]
s8_inds = [i for i, s in enumerate(all_files) if s.startswith('s')]
Om_files = all_files[Om_inds]
s8_files = all_files[s8_inds]

Cls = np.zeros((len(fid_files), 2*(lmax+1)), dtype = np.float64)
for i in np.arange(len(fid_files)):
    Cls[i] = np.append(np.load(fid_dir + fid_files[i], allow_pickle=True).item()[survey+'_low_z'][0:lmax+1], np.load(fid_dir + fid_files[i], allow_pickle=True).item()[survey+'_high_z'][0:lmax+1])

N = len(Cls)
m = numBins
    
C = np.atleast_2d(np.cov(Cls.T)) * (N-1) / (N-m-2)
max_C = np.max(C)
C /= max_C

partial_Om = np.zeros((len(Om_files), 2*(lmax+1)), dtype = np.float64)
for i in np.arange(len(Om_files)):
    partial_Om[i] = np.append(np.load(deriv_dir + Om_files[i], allow_pickle=True).item()[survey+'_low_z'][0:lmax+1], np.load(deriv_dir + Om_files[i], allow_pickle=True).item()[survey+'_high_z'][0:lmax+1])
partial_Om = np.atleast_2d(np.average(partial_Om, axis = 0))

partial_s8 = np.zeros((len(s8_files), 2*(lmax+1)), dtype = np.float64)
for i in np.arange(len(s8_files)):
    partial_s8[i] = np.append(np.load(deriv_dir + s8_files[i], allow_pickle=True).item()[survey+'_low_z'][0:lmax+1], np.load(deriv_dir + s8_files[i], allow_pickle=True).item()[survey+'_high_z'][0:lmax+1])
partial_s8 = np.atleast_2d(np.average(partial_s8, axis = 0))

D = np.append(partial_Om, partial_s8, axis = 0).T
D /= np.sqrt(max_C)


def det_fish(mat):
    return np.linalg.det(D.T @ mat @ np.linalg.inv(mat.T @ C @ mat) @ mat.T @ D)


XL, XH = 0, len(Cls[0])-1             # x–range of the tophat box
YL, YH = 0, numBins-1        # y–range of the tophat box

def f(x1, y1, y2, x3, y3):
    x2=lmax+1 # pivot at the shift from low to high z.
    return np.log(det_fish(create_linlin_matrix([[x1, y1], [x2, y2], [x3, y3]], XH)))

def log_prior(theta):
    x1, y1, y2, x3, y3 = theta
    x2 = lmax+1
    in_box = (XL < x1 < XH) and (XL < x3 < XH) and (YL < y1 < YH) and (YL < y2 < YH) and (YL < y3 < YH)
    ordered = (x3>x2) and (x2>x1) and (y3>y2) and (y2>y1)
    not_singular = (y1<x1) and (y2<x2) and (y3<x3) and (y1-YH>x1-XH) and (y2-YH>x2-XH) and (y3-YH>x3-XH) and (np.abs(y1-y2) < np.abs(x1-x2)) and (np.abs(y2-y3) < np.abs(x2-x3))
    return 0.0 if (in_box and ordered and not_singular) else -np.inf   # 0 = log(1)

def log_likelihood(theta):
    # Treat f() as log-likelihood; scale by temperature later if you like
    return f(*theta)

def log_posterior(theta):
    lp = log_prior(theta)
    if np.isfinite(lp):
        return lp + log_likelihood(theta)
    else:
        return -np.inf


ndim      = 5
nwalkers  = 32
nsteps    = 20000          # long enough for several autocorrelation times
burn_in   = 5000

# Initial walkers: sprinkle them randomly **inside the allowed region**
rng = np.random.default_rng()#42)
p0  = []
while len(p0) < nwalkers:
    trial = rng.uniform(low=[XL, YL, YL, 50, YL], high=[50, YH, YH, XH, YH])
    if log_prior(trial) > -np.inf:
        p0.append(trial)


# --------------------------------------------------------------------------
# 2.  Create a disk-backed backend  – will append new samples if the file
#     exists, or (re-)start if you pass overwrite=True.
backend = emcee.backends.HDFBackend(write_file, read_only=False) # , overwrite=True)


with mp.Pool(processes=mp.cpu_count()) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        pool=pool,              # <- parallel likelihoods  :contentReference[oaicite:0]{index=0}
        backend=backend         # <- incremental on-disk storage   :contentReference[oaicite:1]{index=1}
    )
    
    # --- burn-in -----------------------------------------------------------
    print("Running burn-in …")
    state = sampler.run_mcmc(p0, burn_in, progress=True)

    # --- main production chain --------------------------------------------
    sampler.reset()                             # optional: keep if you
                                                # want burn-in discarded
    print("Production …")
    sampler.run_mcmc(state, nsteps, progress=True)

"""
Utilities for cutting HEALPix maps into patches.
"""

import numpy as np
import healpy as hp
from mpi4py import MPI
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

def index2radec(index, nside=2048, nest=False):
    """
    Converts HEALPix index to Declination and Right Ascension.

    Args:
        index (array): HEALPix pixel indices.
        nside (int): HEALPix nside parameter.
        nest (bool, optional): Nesting scheme of the HEALPix pixels. Defaults to False.

    Returns:
        tuple: Declination and Right Ascension.
    """
    theta, phi = hp.pixelfunc.pix2ang(nside, index, nest=nest)
    return -np.degrees(theta - np.pi / 2.), np.degrees(phi)

def radec2index(ra, dec, nside=2048, nest=False):
    """
    Converts RA, DEC to HEALPix indices.

    Args:
        ra (float or array): Right Ascension in degrees.
        dec (float or array): Declination in degrees.
        nside (int, optional): HEALPix nside parameter. Defaults to 2048.

    Returns:
        int or array: HEALPix pixel indices.
    """
    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    return pix

def preprocess_healpix_map(map_data, nside_highres, nside_lowres, down_grade_to_nside=None, 
                          mask_highres=None, area_threshold=0.2):
    """
    Preprocess HEALPix map: downgrade if requested, setup masks, and compute indices.
    
    Args:
        map_data: Input HEALPix map
        nside_highres: Original high resolution nside
        nside_lowres: Target low resolution nside
        down_grade_to_nside: Optional nside to downgrade map to
        mask_highres: Optional high resolution mask
        area_threshold: Threshold for valid pixels
        
    Returns:
        dict: Preprocessed data including map, masks, and indices
    """
    # Downgrade map if requested
    if down_grade_to_nside is not None:
        map_data = hp.ud_grade(map_data, nside_out=down_grade_to_nside)
        nside_highres = down_grade_to_nside

    # Initialize mask if not provided
    if mask_highres is None:
        mask_highres = np.ones(hp.nside2npix(nside_highres))
    elif down_grade_to_nside is not None:
        mask_highres = hp.ud_grade(mask_highres, nside_out=down_grade_to_nside)

    # Create reference maps for masking
    N_highrespix_in_lowres = hp.nside2npix(nside_highres) / hp.nside2npix(nside_lowres)
    occupancy_map = hp.ud_grade(mask_highres, nside_out=nside_lowres, power=-2)
    mask_lowres = occupancy_map / N_highrespix_in_lowres > area_threshold

    # Get pixel indices and coordinates
    map_indexes_highres = np.arange(hp.nside2npix(nside_highres))
    map_indexes_lowres = np.arange(hp.nside2npix(nside_lowres))

    dec_highres, ra_highres = index2radec(map_indexes_highres, nside_highres)
    dec_lowres, ra_lowres = index2radec(map_indexes_lowres, nside_lowres)
    highres_pix_in_lowres_index = radec2index(ra_highres, dec_highres, nside=nside_lowres)

    # Calculate patch parameters
    delta = np.sqrt(hp.nside2pixarea(nside_lowres, degrees=True)) * 4
    res = hp.nside2resol(nside_highres, arcmin=True)
    pixels = int(delta / (res / 60))
    xsize = 2**int(np.log2(pixels))

    return {
        'map_data': map_data,
        'mask_highres': mask_highres,
        'mask_lowres': mask_lowres,
        'nside_highres': nside_highres,
        'ra_lowres': ra_lowres,
        'dec_lowres': dec_lowres,
        'highres_pix_in_lowres_index': highres_pix_in_lowres_index,
        'xsize': xsize,
        'res': res
    }

def cut_patches(map_data, nside_highres=2048, nside_lowres=16, area_threshold=0.2, 
               mask_highres=None, down_grade_to_nside=None, comm=None, rank=0, size=1):
    """
    Cut a HEALPix map into square patches using gnomonic projection.
    All workers process different patches but load the same map.
    
    Args:
        map_data : array-like
            Input HEALPix map, in nside_highres resolution
        nside_highres : int
            Resolution of input map
        nside_lowres : int
            Resolution for the large pixel in low healpix resolution
        area_threshold : float
            Minimum fraction of occupied area to keep a lowres pixel
        mask_highres : array-like, optional
            HEALPix mask at nside_highres resolution
        down_grade_to_nside : int, optional
            If not None, downgrade the map to this nside before cutting patches
        comm : MPI communicator, optional
            MPI communicator (default: None)
        rank : int, optional
            MPI rank (default: 0)
        size : int, optional
            Number of MPI processes (default: 1)
    """
    # Print initial status
    if rank == 0:
        print("[cut_patches] Starting patch cutting")
        if comm is not None:
            print(f"[cut_patches] Using {size} MPI processes")

    # Preprocess data on rank 0
    if rank == 0:
        processed_data = preprocess_healpix_map(
            map_data, nside_highres, nside_lowres, 
            down_grade_to_nside, mask_highres, area_threshold
        )
    else:
        processed_data = None

    # Broadcast processed data to all ranks
    if comm is not None:
        processed_data = comm.bcast(processed_data, root=0)

    # Extract processed data
    map_data = processed_data['map_data']
    mask_lowres = processed_data['mask_lowres']
    ra_lowres = processed_data['ra_lowres']
    dec_lowres = processed_data['dec_lowres']
    highres_pix_in_lowres_index = processed_data['highres_pix_in_lowres_index']
    xsize = processed_data['xsize']
    res = processed_data['res']

    # Get valid low-res pixels
    valid_pixels = np.where(mask_lowres)[0]
    total_valid = len(valid_pixels)

    if total_valid == 0:
        if rank == 0:
            print("[cut_patches] No valid pixels found!")
        return None, None

    # Process patches
    local_patches = []
    local_coords = []

    # Setup progress bar on rank 0
    if rank == 0:
        pbar = tqdm(total=total_valid, desc="Cutting patches", 
                   position=0, leave=True, 
                   ncols=80,  # Fixed width
                   mininterval=1.0)  # Update at most once per second

    # Simple work distribution: each process takes every size-th pixel
    for idx in range(rank, total_valid, size):
        i_ = valid_pixels[idx]
        ra_ = ra_lowres[i_]
        dec_ = dec_lowres[i_]

        # Mask the map to show only current low-res pixel
        map_masked = map_data.copy()
        mask_ = highres_pix_in_lowres_index == i_
        map_masked[~mask_] = 0.

        # Create patch using gnomonic projection
        patch = hp.gnomview(
            map_masked,
            rot=(ra_, dec_),
            xsize=xsize,
            reso=res,
            no_plot=True,
            return_projected_map=True
        )
        
        local_patches.append(patch.data)
        local_coords.append((ra_, dec_))

        # Update progress bar on rank 0
        if rank == 0:
            pbar.update(size)
            sys.stdout.flush()

    # Close progress bar on rank 0
    if rank == 0:
        pbar.close()
        sys.stdout.flush()

    # Ensure all processes are synchronized
    if comm is not None:
        comm.Barrier()

    # Gather results from all processes if using MPI
    if comm is not None:
        all_patches = comm.gather(local_patches, root=0)
        all_coords = comm.gather(local_coords, root=0)

        if rank == 0:
            patches = [p for sublist in all_patches for p in sublist]
            coords = [c for sublist in all_coords for c in sublist]
            return patches, coords
        return None, None
    else:
        return local_patches, local_coords
    

def plot_patches(patches, coords, nrows=3, ncols=3, start_idx=0, figsize=(12, 12)):
    """
    Plot patches in a grid layout with a common colorbar
    
    Parameters:
    -----------
    patches : ndarray
        Array of patches to plot
    coords : ndarray
        Array of coordinates (RA, Dec) for each patch
    nrows, ncols : int
        Number of rows and columns in the grid
    start_idx : int
        Starting index for patches to plot
    figsize : tuple
        Figure size in inches
    """
    if len(patches) == 0:
        print("[plot_patches] No patches to plot.")
        return

    # Convert zeros to NaN for better visualization
    patches_nan = np.array([np.where(patch == 0, np.nan, patch) for patch in patches])

    # Create figure and axes grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols + 1,  # +1 for colorbar
                         width_ratios=[1]*ncols + [0.05],  # colorbar width ratio
                         hspace=0, wspace=0)

    # Find global min/max for consistent colorbar
    vmin = np.nanmin(patches_nan[start_idx:start_idx + nrows*ncols])
    vmax = np.nanmax(patches_nan[start_idx:start_idx + nrows*ncols])

    # Plot patches
    axes = []
    for i in range(nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

        if i + start_idx < len(patches):
            im = ax.imshow(patches_nan[i + start_idx], origin='lower',
                         vmin=vmin, vmax=vmax)
            
            # Add text at the top center of each patch
            ra, dec = coords[i + start_idx]
            ax.text(0.5, 0.95, f'(RA, Dec) = ({ra:.1f}°, {dec:.1f}°)',
                   horizontalalignment='center',
                   verticalalignment='top',
                   transform=ax.transAxes,
                   fontsize=8,
                   color='white',
                   bbox=dict(facecolor='black', alpha=0.5, pad=2))
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Add common colorbar
    cax = fig.add_subplot(gs[:, -1])
    plt.colorbar(im, cax=cax, label='µK')

    plt.show()
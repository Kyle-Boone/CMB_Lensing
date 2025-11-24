import numpy as np
import healpy as hp
from patchify import *

nside = 1024
patch_file = '/n/home09/kboone/software/Data/Patches/patches_nside_' + str(nside) + '.npy'
coord_file = '/n/home09/kboone/software/Data/Patches/coords_nside_' + str(nside) + '.npy'
survey_map = np.arange(12*(nside**2)) + 1

patches, coords = cut_patches(
            map_data=survey_map,
            nside_highres=nside,
            nside_lowres=8,
            area_threshold=0.2)
patches = np.array(patches)
coords = np.array(coords)

if np.max(np.abs((patches%1))) < 0.01:
    patch_inds = (patches+0.01).astype(int) - 1
    np.save(patch_file, patch_inds)
    np.save(coord_file, coords)
    
import numpy as np
import torch
import torch.fft
from MorletKernel import FiltersSet

class Lin_Scattering2d(object):
    def __init__(
        self, M, N, S=None, L=4, minS=1, numS=2, 
        precision='single', lin=False,
        l_oversampling=1, frequency_factor=1
    ):
        '''
        M: int (positive)
            the number of pixels along x direction
        N: int (positive)
            the number of pixels along y direction
        S: float (positive)
            the maximum scale. Different meaning for dyadic or linear.
            for dyadic it's the number of dyadic scales used.
            at most int(log2(min(M,N))) - 2 in dyadic case (end is inclusive in this case).
            in linear case it's the maximum "scale" to cover.
            fundamental frequency peak would be with 3*min(M,N)/8
        L: int (positive)
            the number of orientations used for oriented wavelets; 
            or the number of harmonics used for harmonic wavelets (L=1 means only monopole is used).
        minS: float (positive)
            minimum scale, 1 corresponds roughly to Nyquist frequency
            only matters in linear spacing
        numS: int (positive, at least 2)
            number of scales to use
        precision: str ('single' or 'double')
        lin: bool
            whether to use linear spacing (True) or dyadic (False)
        '''
        
        if not lin:
            if S is None:
                S = int(np.log2(min(M,N))) - 2
            scales = np.geomspace(minS, 2**S, numS)
        else:
            if S is None:
                S = 3*min(M,N)/8
            scales = 1/np.linspace(1/minS, 1/S, numS)
        
        filters_set = FiltersSet(M=M, N=N, lin=lin, S=S, L=L, minS=minS, numS=numS).generate_wavelets(precision=precision, 
                    l_oversampling=l_oversampling, 
                    frequency_factor=frequency_factor)
        
        self.scales = scales
        self.num_scales = len(scales)
        self.M, self.N = M, N
        self.L = L
        self.frequency_factor = frequency_factor
        self.l_oversampling = l_oversampling
        self.precision = precision
        
        # filters set in arrays
        dtype = filters_set['psi'][0][0].dtype
        self.filters_set = torch.zeros((self.num_scales,self.L,self.M,self.N), dtype=dtype)
        if len(filters_set['psi'][0]) == 1:
            for s in range(self.num_scales):
                for l in range(L):
                    self.filters_set[s,l] = filters_set['psi'][s*L+l][0]
        else:
            self.filters_set = filters_set['psi']
    


    # ---------------------------------------------------------------------------
    #
    # scattering coefficients (mean of scattering fields) without synthesis
    #
    # ---------------------------------------------------------------------------
    def scattering_coef_simple(
        self, data, s1s2_criteria='s2>=s1', 
        pseudo_coef=1, 
    ):
        M, N, scales, L = self.M, self.N, self.scales, self.L
        num_scales = len(scales)
        N_image = data.shape[0]
        filters_set = self.filters_set
        # Deleted optional weights
        # These would be predefined, not like a mask.

        # convert numpy array input into torch tensors
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)

        # initialize tensors for scattering coefficients
        S0 = torch.zeros((N_image,1), dtype=data.dtype)
        P00= torch.zeros((N_image,num_scales,L), dtype=data.dtype)
        S1 = torch.zeros((N_image,num_scales,L), dtype=data.dtype)
        S2 = torch.zeros((N_image,num_scales,num_scales,L,L), dtype=data.dtype) + np.nan
        S2_iso = torch.zeros((N_image,num_scales,num_scales,L), dtype=data.dtype)
        
        # deleted gpu compatability
        
        # 0th order
        S0[:,0] = data.mean((-2,-1))
        
        # 1st and 2nd order
        data_f = torch.fft.fftn(data, dim=(-2,-1))

        # only use the low-k Fourier coefs when calculating large-j scattering coefs.
        for s_num_1 in np.arange(num_scales):
            s1 = scales[s_num_1]
            # 1st order: cut high k
            dx1, dy1 = self.get_dxdy(s1)
            data_f_small = cut_high_k_off(data_f, dx1, dy1)
            wavelet_f = cut_high_k_off(filters_set[s_num_1], dx1, dy1)
            _, M1, N1 = wavelet_f.shape
            # print('M1' + str(M1))
            # I1(j1, l1) = | I * psi(j1, l1) |, "*" means convolution
            I1 = torch.fft.ifftn(
                data_f_small[:,None,:,:] * wavelet_f[None,:,:,:],
                dim=(-2,-1),
            ).abs()
            # S1 = I1 averaged over (x,y)
            S1 [:,s_num_1] = (I1**pseudo_coef).mean((-2,-1)) * M1*N1/M/N
            P00[:,s_num_1] = (I1**2).mean((-2,-1)) * (M1*N1/M/N)**2
            # 2nd order
            I1_f = torch.fft.fftn(I1, dim=(-2,-1))
            del I1
            for s_num_2 in np.arange(num_scales):
                s2 = scales[s_num_2]
                if eval(s1s2_criteria):
                    # cut high k
                    dx2, dy2 = self.get_dxdy(s2)
                    I1_f_small = cut_high_k_off(I1_f, dx2, dy2)
                    wavelet_f2 = cut_high_k_off(filters_set[s_num_2], dx2, dy2)
                    _, M2, N2 = wavelet_f2.shape
                    # I1(j1, l1, j2, l2) = | I1(j1, l1) * psi(j2, l2) |
                    #                    = || I * psi(j1, l1) | * psi(j2, l2)| 
                    # "*" means convolution
                    I2 = torch.fft.ifftn(
                        I1_f_small[:,:,None,:,:] * wavelet_f2[None,None,:,:,:], 
                        dim=(-2,-1),
                    ).abs()
                    # S2 = I2 averaged over (x,y)
                    S2[:,s_num_1,s_num_2,:,:] = (
                        I2**pseudo_coef
                    ).mean((-2,-1)) * M2*N2/M/N
        # Deleted code that could do each l individually, slower but saves memory.

        # average over l1
        S1_iso =  S1.mean(-1)
        P00_iso= P00.mean(-1)
        for l1 in range(L):
            for l2 in range(L):
                S2_iso [:,:,:,(l2-l1)%L] += S2 [:,:,:,l1,l2]
        S2_iso  /= L
        
        # define two reduced s2 coefficients
        s21 = S2_iso.mean(-1) / S1_iso[:,:,None]
        s22 = S2_iso[:,:,:,0] / S2_iso[:,:,:,L//2]
        
        return {'S0':S0,  
                'S1':S1,
                'S2':S2,
                'S1_iso':  S1_iso, 
                'S2_iso':  S2_iso, 's21':s21, 's22':s22,
                'P00_iso':P00_iso,
        }
       
    
    # ---------------------------------------------------------------------------
    #
    # utility functions for computing scattering coef and covariance
    #
    # ---------------------------------------------------------------------------
     
    def get_dxdy(self, s):
        # Replaced any 2**j with an s for the scale
        # Could be useful to think about this in more detail
        dx = int(max( 8, min( np.ceil(self.M/s*self.frequency_factor), self.M//2 ) ))
        dy = int(max( 8, min( np.ceil(self.N/s*self.frequency_factor), self.N//2 ) ))
        return dx, dy

    
# ------------------------------------------------------------------------------------------
#
# end of scattering calculator
#
# ------------------------------------------------------------------------------------------




# ------------------------------------------------------------------------------------------
#
# utility functions 
#
# ------------------------------------------------------------------------------------------
def cut_high_k_off(data_f, dx, dy):
    # Check if this still works for non power of two dx dy values (it should)
    if_xodd = (data_f.shape[-2]%2==1)
    if_yodd = (data_f.shape[-1]%2==1)
    result = torch.cat(
        (torch.cat(
            ( data_f[...,:dx+if_xodd, :dy+if_yodd] , data_f[...,-dx:, :dy+if_yodd]
            ), -2),
          torch.cat(
            ( data_f[...,:dx+if_xodd, -dy:] , data_f[...,-dx:, -dy:]
            ), -2)
        ),-1)
    return result

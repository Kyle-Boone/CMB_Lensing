import numpy as np
import torch
import torch.fft

class FiltersSet(object):
    def __init__(self, M, N, lin=False, S=None, L=4, minS=1, numS=2):
        # Specify dyadic or linear scaling.
        # If dyadic scaling, S is interpreted like J. 
        # minS and numS only used for a linspace in the case of linear scaling
        if not lin:
            if S is None:
                S = int(np.log2(min(M,N))) - 2
            scales = np.geomspace(minS, 2**S, numS)
        else:
            if S is None:
                S = 3*min(M,N)/8
            scales = 1/np.linspace(1/minS, 1/S, numS)
        self.M = M
        self.N = N
        self.scales = scales
        self.num_scales = len(scales)
        self.L = L
        
    def generate_wavelets(
        self, if_save=False, save_dir=None, 
        wavelets='morlet', precision='single', 
        l_oversampling=1, frequency_factor=1
    ):
        # Morlet Wavelets
        if precision=='double':
            dtype = torch.float64
            dtype_np = np.float64
        if precision=='single':
            dtype = torch.float32
            dtype_np = np.float32
        if precision=='half':
            dtype = torch.float16
            dtype_np = np.float16
            
        scales = self.scales

        psi = torch.zeros((self.num_scales, self.L, self.M, self.N), dtype=dtype)
        for num_s in np.arange(self.num_scales):
            s = scales[num_s]
            for l in range(self.L):
                k0 = frequency_factor * 3.0 / 4.0 * np.pi / s
                theta0 = (int(self.L-self.L/2-1)-l) * np.pi / self.L
                
                if wavelets=='morlet':
                    wavelet_spatial = self.morlet_2d(
                        M=self.M, N=self.N, xi=k0, theta=theta0,
                        sigma=0.8 * s / frequency_factor, 
                        slant=4.0 / self.L * l_oversampling,
                    )
                    wavelet_Fourier = np.fft.fft2(wavelet_spatial)
                
                wavelet_Fourier[0,0] = 0
                psi[num_s, l] = torch.from_numpy(wavelet_Fourier.real.astype(dtype_np))
        
            
        filters_set = {'psi':psi}
        if if_save:
            np.save(
                save_dir + 'filters_set_M' + str(self.M) + 'N' + str(self.N)
                + 'NumScales' + str(self.num_scales) + 'L' + str(self.L) + '_' + precision + '.npy', 
                np.array([{'filters_set': filters_set}])
            )
        return filters_set
    
    # Morlet Wavelets
    def morlet_2d(self, M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=False):
        """
            (from kymatio package) 
            Computes a 2D Morlet filter.
            A Morlet filter is the sum of a Gabor filter and a low-pass filter
            to ensure that the sum has exactly zero mean in the temporal domain.
            It is defined by the following formula in space:
            psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
            where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
            the cancelling parameter.
            Parameters
            ----------
            M, N : int
                spatial sizes
            sigma : float
                bandwidth parameter
            xi : float
                central frequency (in [0, 1])
            theta : float
                angle in [0, pi]
            slant : float, optional
                parameter which guides the elipsoidal shape of the morlet
            offset : int, optional
                offset by which the signal starts
            fft_shift : boolean
                if true, shift the signal in a numpy style
            Returns
            -------
            morlet_fft : ndarray
                numpy array of size (M, N)
        """
        wv = self.gabor_2d_mycode(M, N, sigma, theta, xi, slant, offset, fft_shift)
        wv_modulus = self.gabor_2d_mycode(M, N, sigma, theta, 0, slant, offset, fft_shift)
        K = wv.sum() / wv_modulus.sum()

        mor = wv - K * wv_modulus
        return mor

    def gabor_2d_mycode(self, M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=False):
        """
            (partly from kymatio package)
            Computes a 2D Gabor filter.
            A Gabor filter is defined by the following formula in space:
            psi(u) = g_{sigma}(u) e^(i xi^T u)
            where g_{sigma} is a Gaussian envelope and xi is a frequency.
            Parameters
            ----------
            M, N : int
                spatial sizes
            sigma : float
                bandwidth parameter
            xi : float
                central frequency (in [0, 1])
            theta : float
                angle in [0, pi]
            slant : float, optional
                parameter which guides the elipsoidal shape of the morlet
            offset : int, optional
                offset by which the signal starts
            fft_shift : boolean
                if true, shift the signal in a numpy style
            Returns
            -------
            morlet_fft : ndarray
                numpy array of size (M, N)
        """
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float64)
        R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float64)
        D = np.array([[1, 0], [0, slant * slant]])
        curv = np.matmul(R, np.matmul(D, R_inv)) / ( 2 * sigma * sigma)

        gab = np.zeros((M, N), np.complex128)
        xx = np.empty((2,2, M, N))
        yy = np.empty((2,2, M, N))

        for ii, ex in enumerate([-1, 0]):
            for jj, ey in enumerate([-1, 0]):
                xx[ii,jj], yy[ii,jj] = np.mgrid[
                    offset + ex * M : offset + M + ex * M, 
                    offset + ey * N : offset + N + ey * N
                ]
        
        arg = -(curv[0, 0] * xx * xx + (curv[0, 1] + curv[1, 0]) * xx * yy + curv[1, 1] * yy * yy) +\
            1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
        gab = np.exp(arg).sum((0,1))

        norm_factor = 2 * np.pi * sigma * sigma / slant
        gab = gab / norm_factor

        if fft_shift:
            gab = np.fft.fftshift(gab, axes=(0, 1))
        return gab

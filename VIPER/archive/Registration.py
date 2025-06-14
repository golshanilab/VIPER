import numpy as np
from scipy import ndimage
from datetime import datetime


#%% Smooth plus Periodic Image Decomposition Functions


'''
Subtract the lowest spatial frequency component of the Image by decomposing it into its
periodic and smooth component.

"Periodic Plus Smooth Image Decomposition" Lionel Moisan, Journal of Mathematical Imaging 
and Vision, Vol 39,(161-179) 2011.
'''

def periodic_smooth_decomp(I: np.ndarray) -> (np.ndarray, np.ndarray):
    '''Performs periodic-smooth image decomposition

    Parameters
    ----------
    I : np.ndarray
        [M, N] image. will be coerced to a float.

    Returns
    -------
    P : np.ndarray
        [M, N] image, float. periodic portion.
    S : np.ndarray
        [M, N] image, float. smooth portion.
    '''
    u = I.astype(np.float64)
    v = u2v(u)
    v_fft = np.fft.fftn(v)
    s = v2s(v_fft)
    s_i = np.fft.ifftn(s)
    s_f = np.real(s_i)
    p = u - s_f # u = p + s
    
    return p, s_f

def u2v(u: np.ndarray) -> np.ndarray:
    '''Converts the image `u` into the image `v`

    Parameters
    ----------
    u : np.ndarray
        [M, N] image

    Returns
    -------
    v : np.ndarray
        [M, N] image, zeroed expect for the outermost rows and cols
    '''
    
    v = np.zeros(u.shape, dtype=np.float64)

    v[0, :] = np.subtract(u[-1, :], u[0,  :], dtype=np.float64)
    v[-1,:] = np.subtract(u[0,  :], u[-1, :], dtype=np.float64)

    v[:,  0] += np.subtract(u[:, -1], u[:,  0], dtype=np.float64)
    v[:, -1] += np.subtract(u[:,  0], u[:, -1], dtype=np.float64)
    return v

def v2s(v_hat: np.ndarray) -> np.ndarray:
    '''Computes the maximally smooth component of `u`, `s` from `v`


    s[q, r] = v[q, r] / (2*np.cos( (2*np.pi*q)/M )
        + 2*np.cos( (2*np.pi*r)/N ) - 4)

    Parameters
    ----------
    v_hat : np.ndarray
        [M, N] DFT of v
    '''
    M, N = v_hat.shape

    q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = np.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2*np.cos( np.divide((2*np.pi*q), M) ) \
         + 2*np.cos( np.divide((2*np.pi*r), N) ) - 4)
        
    den[den==0] = 1
    s = np.divide(v_hat, den, out=np.zeros_like(v_hat))
    s[0, 0] = 0
    return s

#%% Fourier Phase Correlation Image Registration Functions

'''
Sub-pixel image registration by Fourier phase correlation and discrete fourier upsampling.

"Efficient Subpixel Image Registration Algorithms" Manuel Guizar-Sicairos, Samuel T. Thurman, 
and James R. Fienup, Optics Letters Vol. 33, Issue 2, (156-158) 2008. 

'''

def dft_ups(inp, nor=None, noc=None, usfac=1, roff=0, coff=0):
    '''
    Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
    a small region.

    This code is intended to provide the same result as if the following
    operations were performed:

      * Embed the array "in" in an array that is usfac times larger in each
        dimension. ifftshift to bring the center of the image to (1,1).
      * Take the FFT of the larger array
      * Extract an [nor, noc] region of the result. Starting with the
        [roff+1 coff+1] element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the
    zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]

    Parameters
    ----------
    usfac : int
        Upsampling factor (default usfac = 1)
    nor,noc : int,int
        Number of pixels in the output upsampled DFT, in units of upsampled
        pixels (default = size(in))
    roff, coff : int, int
        Row and column offsets, allow to shift the output array to a region of
        interest on the DFT (default = 0)
    '''

    nr, nc = np.shape(inp)

    if noc is None:
        noc = nc
    if nor is None:
        nor = nr

    # Compute kernels and obtain DFT by matrix products
    term1c = (np.fft.ifftshift(np.arange(nc, dtype='float')-np.floor(nc/2)).T[:, np.newaxis])  # fftfreq
    term2c = (np.arange(noc, dtype='float') - coff)[np.newaxis,:]              # output points
    kernc = np.exp((-1j*2*np.pi/(nc*usfac))*term1c*term2c)

    term1r = (np.arange(nor, dtype='float').T - roff)[:, np.newaxis]                # output points
    term2r = (np.fft.ifftshift(np.arange(nr, dtype='float'))-np.floor(nr/2))[np.newaxis,:]  # fftfreq
    kernr = np.exp((-1j*2*np.pi/(nr*usfac))*term1r*term2r)
    out = np.dot(np.dot(kernr, inp), kernc)

    return out


def fourier_imgreg(og, temp, im2, NNr, NNc, nnr, nnc, usfac=1, maxoff=5):
    '''
    Image registration by Fourier phase correlation

    Parameters
    ----------
    og : Original Input Image to shift
    temp : Template to match the input image to (Already in fourier)
    im2 : Cropped and processed original image to calculate shift
    NNr : row mesh
    NNc : column mesh
    nnr : number of rows
    nnc : number of columns
    usfac : Upsampling factor
    maxoff : MAximum allowed rigid shift

    Returns
    -------
    registered image and shift

    '''
    ogIm,bgIm = periodic_smooth_decomp(og)
    ogIm = np.fft.fft2(ogIm)

    im2 = periodic_smooth_decomp(im2)[0]
    im2 = im2 - np.average(im2)
    buf1ft = temp
    buf2ft = np.fft.fft2(im2)
    [m, n] = np.shape(buf1ft)

    CClarge = np.zeros((m*2, n*2), dtype='complex')
    CClarge[int(m-np.fix(m/2)):int(m+np.fix((m-1)/2)+1), int(n-np.fix(n/2)):int(n + np.fix((n-1)/2)+1)] = np.fft.fftshift(buf1ft) * np.conj(np.fft.fftshift(buf2ft))
    CC = np.fft.ifft2(np.fft.ifftshift(CClarge))

    CC[maxoff:-maxoff, :] = 0
    CC[:, maxoff:-maxoff] = 0
    rloc, cloc = np.unravel_index(abs(CC).argmax(), CC.shape)
    CCmax = CC[rloc, cloc]

    if rloc > m:
        row_shift2 = rloc - m*2
    else:
        row_shift2 = rloc
    if cloc > n:
        col_shift2 = cloc - n*2
    else:
        col_shift2 = cloc

    row_shift2 = row_shift2/2.
    col_shift2 = col_shift2/2.

    zoom_factor = 1.5
    row_shift0 = np.round(row_shift2*usfac)/usfac
    col_shift0 = np.round(col_shift2*usfac)/usfac
    dftshift = np.floor(np.ceil(usfac*zoom_factor)/2)

    roff = dftshift-row_shift0*usfac
    coff = dftshift-col_shift0*usfac

    upsampled = dft_ups((buf2ft*np.conj(buf1ft)),
                        np.ceil(usfac*zoom_factor),
                        np.ceil(usfac*zoom_factor),
                        usfac,
                        roff,
                        coff)

    CC = np.conj(upsampled)/(m*n*usfac**2)
    rloc, cloc = np.unravel_index(abs(CC).argmax(), CC.shape)
    CCmax = CC[rloc, cloc]

    rg00 = dft_ups(buf1ft * np.conj(buf1ft), 1, 1, usfac)/(m*n*usfac**2)
    rf00 = dft_ups(buf2ft * np.conj(buf2ft), 1, 1, usfac)/(m*n*usfac**2)

    cloc = cloc - dftshift
    rloc = rloc - dftshift
    row_shift = row_shift0 + rloc/usfac
    col_shift = col_shift0 + cloc/usfac

    error = 1.0-CCmax*np.conj(CCmax)/(rg00*rf00)
    error = np.sqrt(error)
    diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))
    
    reg = ogIm*np.exp(1j*2*np.pi*(-row_shift*NNr/nnr-col_shift*NNc/nnc))
    reg = reg*np.exp(1j*diffphase)
    reg = abs(np.fft.ifft2(reg))+bgIm

    
    return reg, [row_shift, col_shift]


#%% Registration

def register(rawData, params, temps=[]):
    '''
    Implement Image Registration by Fourier Phase Correlation

    Parameters
    ----------
    imageData : data to register
    params : parameters set by user [min_row, max_row, min_col, max_col, update_temp_every_#frames, use_#frames_new_temp, pseudocontrast_power_exponent]

    Returns
    -------
    registered : registered array
    maxshift : maximum calculated shift
    redo : if unsatisfied, prompts parameter resetting

    '''
        

    t,n,m = rawData.shape
    
    # Create Mesh
    
    Nr = np.fft.ifftshift(np.linspace(-np.fix(n / 2), np.ceil(n / 2) - 1, n))
    Nc = np.fft.ifftshift(np.linspace(-np.fix(m / 2), np.ceil(m / 2) - 1, m))
    [Nc, Nr] = np.meshgrid(Nc, Nr)

    
    # Preprocess Image Data
    
    w = np.ones((3,1,1))/3
    imageData = np.copy(rawData[:,params[0]:params[1],params[2]:params[3]])
    imageData = imageData/np.amax(imageData,axis=(1,2))[:,np.newaxis,np.newaxis]
    imageData = np.power(imageData,params[6])
    imageData = ndimage.convolve(imageData,w)
    
    # Define Template
    
    tnum = 0
    temp = np.average(imageData[0:params[5]], axis=0)
    if len(temps) == 0:
        temps = [np.copy(temp)]
    else:
        temp = np.median(temps,axis=0)   
    # Preprocess Template
    
    temp = periodic_smooth_decomp(temp)[0]
    temp = temp-np.average(temp)
    temp = np.fft.fft2(temp)
    
    # Initialize Arrays
    
    registered = np.zeros(rawData.shape)
    shifts = np.zeros((t,2))
    
    for i in range(t):
        processed_frame = imageData[i]
        raw_frame = rawData[i]
        if i > 0 and i % params[4] == 0:
            tnum = tnum+1
            new = np.array(registered[i-params[4]+1:i-params[4]+params[5]+1,params[0]:params[1],params[2]:params[3]])
            new = new/np.amax(new,axis=(1,2))[:,np.newaxis,np.newaxis]
            new = np.power(new,params[6])
            new = np.average(new,axis=0)
            if len(temps) > 10:
                temps[tnum] = new
            else:
                temps = np.append(temps,[new],axis=0)
            temp = np.median(temps,axis=0)
            temp = periodic_smooth_decomp(temp)[0]
            temp = temp - np.average(temp)
            temp = np.fft.fft2(temp)
            if tnum == 9:
                tnum = 0
        out = fourier_imgreg(raw_frame, temp, processed_frame, Nr, Nc, n, m, usfac=200, maxoff=200)
        registered[i,:,:] = out[0]
        shifts[i,:] = [out[1][0],out[1][1]]

    return registered, shifts, temps



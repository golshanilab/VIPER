import cupy as cp
from cupyx.scipy import ndimage, signal
from scipy.interpolate import interp1d
from datetime import datetime
import matplotlib.pyplot as plt
import tifffile as tif
import time
import numpy as np
from scipy.optimize import curve_fit

#%% Smooth plus Periodic Image Decomposition Functions


'''
Subtract the lowest spatial frequency component of the Image by decomposing it into its
periodic and smooth component.

"Periodic Plus Smooth Image Decomposition" Lionel Moisan, Journal of Mathematical Imaging 
and Vision, Vol 39,(161-179) 2011.
'''

def periodic_smooth_decomp(I: cp.ndarray) -> (cp.ndarray, cp.ndarray):
    '''Performs periodic-smooth image decomposition

    Parameters
    ----------
    I : cp.ndarray
        [M, N] image. will be coerced to a float.

    Returns
    -------
    P : cp.ndarray
        [M, N] image, float. periodic portion.
    S : cp.ndarray
        [M, N] image, float. smooth portion.
    '''
    u = I.astype(cp.float64)
    v = u2v(u)
    v_fft = cp.fft.fftn(v)
    s = v2s(v_fft)
    s_i = cp.fft.ifftn(s)
    s_f = cp.real(s_i)
    p = u - s_f # u = p + s
    
    return p, s_f

def u2v(u: cp.ndarray) -> cp.ndarray:
    '''Converts the image `u` into the image `v`

    Parameters
    ----------
    u : cp.ndarray
        [M, N] image

    Returns
    -------
    v : cp.ndarray
        [M, N] image, zeroed expect for the outermost rows and cols
    '''
    
    v = cp.zeros(u.shape, dtype=cp.float64)

    v[0, :] = cp.subtract(u[-1, :], u[0,  :], dtype=cp.float64)
    v[-1,:] = cp.subtract(u[0,  :], u[-1, :], dtype=cp.float64)

    v[:,  0] += cp.subtract(u[:, -1], u[:,  0], dtype=cp.float64)
    v[:, -1] += cp.subtract(u[:,  0], u[:, -1], dtype=cp.float64)
    return v

def v2s(v_hat: cp.ndarray) -> cp.ndarray:
    '''Computes the maximally smooth component of `u`, `s` from `v`


    s[q, r] = v[q, r] / (2*cp.cos( (2*cp.pi*q)/M )
        + 2*cp.cos( (2*cp.pi*r)/N ) - 4)

    Parameters
    ----------
    v_hat : cp.ndarray
        [M, N] DFT of v
    '''
    M, N = v_hat.shape

    q = cp.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = cp.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2*cp.cos( cp.divide((2*cp.pi*q), M) ) \
         + 2*cp.cos( cp.divide((2*cp.pi*r), N) ) - 4)
        
    den[den==0] = 1
    s = cp.divide(v_hat, den, out=cp.zeros_like(v_hat))
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

    nr, nc = cp.shape(inp)

    if noc is None:
        noc = nc
    if nor is None:
        nor = nr

    # Compute kernels and obtain DFT by matrix products
    term1c = (cp.fft.ifftshift(cp.arange(nc, dtype='float')-cp.floor(nc/2)).T[:, cp.newaxis])  # fftfreq
    term2c = (cp.arange(noc, dtype='float') - coff)[cp.newaxis,:]              # output points
    kernc = cp.exp((-1j*2*cp.pi/(nc*usfac))*term1c*term2c)

    term1r = (cp.arange(nor, dtype='float').T - roff)[:, cp.newaxis]                # output points
    term2r = (cp.fft.ifftshift(cp.arange(nr, dtype='float'))-cp.floor(nr/2))[cp.newaxis,:]  # fftfreq
    kernr = cp.exp((-1j*2*cp.pi/(nr*usfac))*term1r*term2r)
    out = cp.dot(cp.dot(kernr, inp), kernc)

    return out


def fourier_imgreg(temp, im2, usfac=1, maxoff=5):
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

    im2 = periodic_smooth_decomp(im2)[0]
    im2 = im2 - cp.average(im2)
    buf1ft = temp
    buf2ft = cp.fft.fft2(im2)
    [m, n] = cp.shape(buf1ft)

    CClarge = cp.zeros((m*2, n*2), dtype='complex')
    CClarge[int(m-cp.fix(m/2)):int(m+cp.fix((m-1)/2)+1), int(n-cp.fix(n/2)):int(n + cp.fix((n-1)/2)+1)] = cp.fft.fftshift(buf1ft) * cp.conj(cp.fft.fftshift(buf2ft))
    CC = cp.fft.ifft2(cp.fft.ifftshift(CClarge))

    CC[maxoff:-maxoff, :] = 0
    CC[:, maxoff:-maxoff] = 0
    rloc, cloc = cp.unravel_index(abs(CC).argmax(), CC.shape)

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
    row_shift0 = cp.round_(row_shift2*usfac)/usfac
    col_shift0 = cp.round_(col_shift2*usfac)/usfac
    dftshift = cp.floor(cp.ceil(usfac*zoom_factor)/2)

    roff = dftshift-row_shift0*usfac
    coff = dftshift-col_shift0*usfac

    upsampled = dft_ups((buf2ft*cp.conj(buf1ft)),
                        cp.ceil(usfac*zoom_factor),
                        cp.ceil(usfac*zoom_factor),
                        usfac,
                        roff,
                        coff)

    CC = cp.conj(upsampled)/(m*n*usfac**2)
    rloc, cloc = cp.unravel_index(abs(CC).argmax(), CC.shape)

    cloc = cloc - dftshift
    rloc = rloc - dftshift
    row_shift = row_shift0 + rloc/usfac
    col_shift = col_shift0 + cloc/usfac
    
    return row_shift, col_shift

def apply_shift(im, row_shift, col_shift, NNr, NNc, nnr, nnc):

    ogIm,bgIm = periodic_smooth_decomp(im)
    ogIm = cp.fft.fft2(ogIm)
    reg = ogIm*cp.exp(1j*2*cp.pi*(-row_shift*NNr/nnr-col_shift*NNc/nnc))
    reg = abs(cp.fft.ifft2(reg))+bgIm

    return reg

def sin_fit(x, a, b, c):
    return a*np.sin(b*x +c)

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
    
    # Create Meshgrid
    Nr = cp.fft.ifftshift(cp.linspace(-cp.fix(n / 2), cp.ceil(n / 2) - 1, n))
    Nc = cp.fft.ifftshift(cp.linspace(-cp.fix(m / 2), cp.ceil(m / 2) - 1, m))
    [Nc, Nr] = cp.meshgrid(Nc, Nr)

    
    # Preprocess Image Data
    w = cp.ones((3,1,1))/3
    imageData = cp.copy(rawData[:,params[0]:params[1],params[2]:params[3]])
    imageData = imageData/cp.amax(imageData,axis=(1,2))[:,cp.newaxis,cp.newaxis]
    imageData = cp.power(imageData,params[6])
    imageData = ndimage.convolve(imageData,w)
    
    # Define Template
    tnum = 0
    temp = cp.average(imageData[0:params[5]], axis=0)
    if len(temps) == 0:
        temps = [cp.copy(temp)]
    else:
        temp = cp.median(temps,axis=0) 

    # Preprocess Template
    temp = periodic_smooth_decomp(temp)[0]
    temp = temp-cp.average(temp)
    temp = cp.fft.fft2(temp)

    # Initialize Arrays
    registered = cp.zeros(rawData.shape)
    shifts = cp.zeros((t,2))
    for i in range(t):
        processed_frame = imageData[i]
        raw_frame = rawData[i]
        if i > 0 and i % params[4] == 0:
            tnum = tnum+1
            new = cp.array(registered[i-params[4]+1:i-params[4]+params[5]+1,params[0]:params[1],params[2]:params[3]])
            new = new/cp.amax(new,axis=(1,2))[:,cp.newaxis,cp.newaxis]
            new = cp.power(new,params[6])
            new = cp.average(new,axis=0)
            if len(temps) > 10:
                temps[tnum] = new
            else:
                temps = cp.append(temps,[new],axis=0)
            temp = cp.median(temps,axis=0)
            temp = periodic_smooth_decomp(temp)[0]
            temp = temp - cp.average(temp)
            temp = cp.fft.fft2(temp)
            if tnum == 9:
                tnum = 0
        row, col = fourier_imgreg(temp, processed_frame, usfac=200, maxoff=250)
        out = apply_shift(raw_frame, row, col, Nr, Nc, n, m)
        registered[i,:,:] = out
        shifts[i,:] = cp.array([row,col])

    return registered, cp.asnumpy(shifts), temps


def register_fast(rawData, params, temps=[]):

    t,n,m = rawData.shape   

    Nr = cp.fft.ifftshift(cp.linspace(-cp.fix(n / 2), cp.ceil(n / 2) - 1, n))
    Nc = cp.fft.ifftshift(cp.linspace(-cp.fix(m / 2), cp.ceil(m / 2) - 1, m))
    [Nc, Nr] = cp.meshgrid(Nc, Nr)

    # Define Template
    w = cp.ones((3,1,1))/3
    temp = cp.copy(rawData[0:params[5],params[0]:params[1],params[2]:params[3]])
    temp = temp/cp.amax(temp,axis=(1,2))[:,cp.newaxis,cp.newaxis]
    temp = cp.power(temp,params[6])
    temp = ndimage.convolve(temp,w)

    tnum = 0
    temp = cp.average(temp, axis=0)
    if len(temps) == 0:
        temps = [cp.copy(temp)]
    else:
        temp = cp.median(temps,axis=0) 

    # Preprocess Template
    temp = periodic_smooth_decomp(temp)[0]
    temp = temp-cp.average(temp)
    temp = cp.fft.fft2(temp)

    # Initialize Arrays
    registered = cp.zeros(rawData.shape)
    shifts = cp.zeros((t,2))

    idx = t//params[5]

    # kernel
    past_col_shift = cp.zeros(params[5])
    past_row_shift = cp.zeros(params[5])
    vals_to_interp = np.arange(0,params[5])
    vals_from_intep = np.array([0,params[5]])

    for i in range(idx):

        int_frame = cp.copy(rawData[i*params[5]:(i+1)*params[5],params[0]:params[1],params[2]:params[3]])
        int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
        int_frame = cp.power(int_frame,params[6])
        #int_frame = ndimage.convolve(int_frame,w)
        int_frame = cp.average(int_frame,axis=0)

        if i > 0:
            if i % (params[4]//params[5]) == 0:
                tnum = tnum+1
                new = cp.array(registered[(i*params[5])-params[4]+1:(i*params[5])-params[4]+params[5]+1,params[0]:params[1],params[2]:params[3]])
                #tif.imwrite('test_'+str(int(time.time()*1000))+'.tif',cp.asnumpy(new))
                new = new/cp.amax(new,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                new = cp.power(new,params[6])
                #new = ndimage.convolve(new,w)
                new = cp.average(new,axis=0)
                if len(temps) > 10:
                    temps[tnum] = new
                else:
                    temps = cp.append(temps,[new],axis=0)
                temp = cp.median(temps,axis=0)
                temp = periodic_smooth_decomp(temp)[0]
                temp = temp - cp.average(temp)
                
                temp = cp.fft.fft2(temp)
                if tnum == 9:
                    tnum = 0

            row, col = fourier_imgreg(temp, int_frame, usfac=100, maxoff=100)
            
            rows = cp.asnumpy(cp.array([past_row_shift, row]))
            cols = cp.asnumpy(cp.array([past_col_shift, col]))

            cs_row = interp1d(vals_from_intep, rows, kind='slinear')
            cs_col= interp1d(vals_from_intep, cols, kind='slinear')
            #row_int_shifts = cp.interp(vals_to_interp, vals_from_intep, rows)
            #col_int_shifts = cp.interp(vals_to_interp, vals_from_intep, cols)
            row_int_shifts = cs_row(vals_to_interp)
            col_int_shifts = cs_col(vals_to_interp)

            shifts[(i*params[5]) - (params[5]//2) : (i*params[5]) + (params[5]//2),:] = cp.array([row_int_shifts,col_int_shifts]).T

            for k in range(params[5]):
                frame = rawData[(i*params[5])-(params[5]//2)+k]
                out = apply_shift(frame, row_int_shifts[k], col_int_shifts[k], Nr, Nc, n, m)
                #tif.imwrite('test'+str(int(time.time()*1000))+'.tif',cp.asnumpy(out))
                registered[(i*params[5])-(params[5]//2)+k,:,:] = out

            if i == idx-1:
                for k in range(params[5]-(params[5]//2)): 
                    frame = rawData[(i*params[5])+k+(params[5]//2)]
                    out = apply_shift(frame, row, col, Nr, Nc, n, m)
                    #tif.imwrite('test'+str(int(time.time()*1000))+'.tif',cp.asnumpy(out))
                    registered[(i*params[5])+k+(params[5]//2),:,:] = out
                
            
            past_col_shift = col
            past_row_shift = row


        else:
            row, col = fourier_imgreg(temp, int_frame, usfac=100, maxoff=100)
            shifts[0:params[5] - params[5]//2,:] = cp.array([row,col])
            past_col_shift = col
            past_row_shift = row

            for k in range(params[5]-(params[5]//2)): 
                frame = rawData[(i*params[5])+k]
                out = apply_shift(frame, row, col, Nr, Nc, n, m)
                #tif.imwrite('test'+str(int(time.time()*1000))+'.tif',cp.asnumpy(out))
                registered[(i*params[5])+k,:,:] = out



        

    return registered, cp.asnumpy(shifts), temps


def register_faster(rawData, params, temps=[]):

    usfac = 100
    maxoff = 100

    t,n,m = rawData.shape   

    Nr = cp.fft.ifftshift(cp.linspace(-cp.fix(n / 2), cp.ceil(n / 2) - 1, n))
    Nc = cp.fft.ifftshift(cp.linspace(-cp.fix(m / 2), cp.ceil(m / 2) - 1, m))
    [Nc, Nr] = cp.meshgrid(Nc, Nr)

    # Define Template
    w = cp.ones((3,1,1))/3
    temp = cp.copy(rawData[0:params[5],params[0]:params[1],params[2]:params[3]])
    temp = temp/cp.amax(temp,axis=(1,2))[:,cp.newaxis,cp.newaxis]
    temp = cp.power(temp,params[6])
    temp = ndimage.convolve(temp,w)

    tnum = 0
    temp = cp.average(temp, axis=0)
    if len(temps) == 0:
        temps = [cp.copy(temp)]
    else:
        temp = cp.median(temps,axis=0) 

    # Preprocess Template
    temp = periodic_smooth_decomp(temp)[0]
    temp = temp-cp.average(temp)
    temp = cp.fft.fft2(temp)

    # Initialize Arrays
    registered = cp.zeros(rawData.shape)
    shifts = np.zeros((t,2))
    shift_hist = np.zeros((params[4]//params[5],2))
    shift_idx = np.arange(0,params[4])
    calc_shift_idx = np.arange(params[5]//2,params[4]+(params[5]//2),params[5])

    idx = t//params[5]

    for i in range(idx):

        int_frame = cp.copy(rawData[i*params[5]:(i+1)*params[5],params[0]:params[1],params[2]:params[3]])
        int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
        int_frame = cp.power(int_frame,params[6])
        #int_frame = ndimage.convolve(int_frame,w)
        int_frame = cp.average(int_frame,axis=0)

        if i > 0:
            if i % (params[4]//params[5]) == 0:
                # Calculate Interpolation Endpoints
                # Beginning
                int_frame = cp.copy(rawData[(i*params[5])-params[4]:(i*params[5])-params[4]+4,params[0]:params[1],params[2]:params[3]])
                int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                int_frame = cp.power(int_frame,params[6])
                int_frame = cp.average(int_frame,axis=0)
                first_row, first_col = fourier_imgreg(temp, int_frame, usfac=usfac, maxoff=maxoff)

                # End
                int_frame = cp.copy(rawData[(i*params[5])-4:(i*params[5]),params[0]:params[1],params[2]:params[3]])
                int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                int_frame = cp.power(int_frame,params[6])
                int_frame = cp.average(int_frame,axis=0)
                last_row, last_col = fourier_imgreg(temp, int_frame, usfac=usfac, maxoff=maxoff)

                # Initialize Interpolation
                shift_idxs = np.insert(calc_shift_idx,0,0)
                shift_idxs = np.append(shift_idxs,params[4])

                row_shifts = np.copy(shift_hist[:,0][::-1])
                row_shifts = np.insert(row_shifts,0,cp.asnumpy(first_row))
                row_shifts = np.append(row_shifts,cp.asnumpy(last_row))

                col_shifts = np.copy(shift_hist[:,1][::-1])
                col_shifts = np.insert(col_shifts,0,cp.asnumpy(first_col))
                col_shifts = np.append(col_shifts,cp.asnumpy(last_col))

                # Interpolate Shifts
                cs_row = interp1d(shift_idxs, row_shifts, kind='cubic')
                cs_col = interp1d(shift_idxs, col_shifts, kind='cubic')

                local_row_shifts = cs_row(shift_idx)
                local_col_shifts = cs_col(shift_idx)

                shifts[(i*params[5])-params[4]:(i*params[5]),0] = local_row_shifts
                shifts[(i*params[5])-params[4]:(i*params[5]),1] = local_col_shifts

                #shifts[(i*params[5])-params[4]:(i*params[5])+(params[5]//2)-params[4],1] = shift_hist[-1,1]

                #shifts[(i*params[5])-params[4]+(params[5]//2):(i*params[5])-(params[5]//2),0] = local_row_shifts
                #shifts[(i*params[5])-params[4]+(params[5]//2):(i*params[5])-(params[5]//2),1] = local_col_shifts

                #shifts[(i*params[5])-(params[5]//2):(i*params[5]),0] = shift_hist[0,0]
                #shifts[(i*params[5])-(params[5]//2):(i*params[5]),1] = shift_hist[0,1]

                #Apply Shifts
                for k in range(params[4]):
                    frame = rawData[(i*params[5])-params[4]+k]
                    row_shift = shifts[(i*params[5])-params[4]+k,0]
                    col_shift = shifts[(i*params[5])-params[4]+k,1]
                    out = apply_shift(frame, row_shift, col_shift, Nr, Nc, n, m)
                    registered[(i*params[5])-params[4]+k,:,:] = out
                
                # Update Template
                tnum = tnum+1
                new = cp.array(registered[(i*params[5])-params[4]+1:(i*params[5])-params[4]+params[5]+1,params[0]:params[1],params[2]:params[3]])
                new = new/cp.amax(new,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                new = cp.power(new,params[6])
                new = cp.average(new,axis=0)
                if len(temps) > 10:
                    temps[tnum] = new
                else:
                    temps = cp.append(temps,[new],axis=0)
                temp = cp.median(temps,axis=0)
                temp = periodic_smooth_decomp(temp)[0]
                temp = temp - cp.average(temp)
                
                temp = cp.fft.fft2(temp)
                if tnum == 9:
                    tnum = 0
            
            elif i == idx-1:
                i = i+1
                # Calculate Interpolation Endpoints
                # Beginning
                int_frame = cp.copy(rawData[(i*params[5])-params[4]:(i*params[5])-params[4]+4,params[0]:params[1],params[2]:params[3]])
                int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                int_frame = cp.power(int_frame,params[6])
                int_frame = cp.average(int_frame,axis=0)
                first_row, first_col = fourier_imgreg(temp, int_frame, usfac=usfac, maxoff=maxoff)

                # End
                int_frame = cp.copy(rawData[(i*params[5])-4:(i*params[5]),params[0]:params[1],params[2]:params[3]])
                int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                int_frame = cp.power(int_frame,params[6])
                int_frame = cp.average(int_frame,axis=0)
                last_row, last_col = fourier_imgreg(temp, int_frame, usfac=usfac, maxoff=maxoff)

                # Initialize Interpolation
                shift_idxs = np.insert(calc_shift_idx,0,0)
                shift_idxs = np.append(shift_idxs,params[4])

                row_shifts = np.copy(shift_hist[:,0][::-1])
                row_shifts = np.insert(row_shifts,0,cp.asnumpy(first_row))
                row_shifts = np.append(row_shifts,cp.asnumpy(last_row))

                col_shifts = np.copy(shift_hist[:,1][::-1])
                col_shifts = np.insert(col_shifts,0,cp.asnumpy(first_col))
                col_shifts = np.append(col_shifts,cp.asnumpy(last_col))

                # Interpolate Shifts
                cs_row = interp1d(shift_idxs, row_shifts, kind='cubic')
                cs_col = interp1d(shift_idxs, col_shifts, kind='cubic')

                local_row_shifts = cs_row(shift_idx)
                local_col_shifts = cs_col(shift_idx)

                shifts[(i*params[5])-params[4]:(i*params[5]),0] = local_row_shifts
                shifts[(i*params[5])-params[4]:(i*params[5]),1] = local_col_shifts

                #shifts[(i*params[5])-params[4]:(i*params[5])+(params[5]//2)-params[4],1] = shift_hist[-1,1]

                #shifts[(i*params[5])-params[4]+(params[5]//2):(i*params[5])-(params[5]//2),0] = local_row_shifts
                #shifts[(i*params[5])-params[4]+(params[5]//2):(i*params[5])-(params[5]//2),1] = local_col_shifts

                #shifts[(i*params[5])-(params[5]//2):(i*params[5]),0] = shift_hist[0,0]
                #shifts[(i*params[5])-(params[5]//2):(i*params[5]),1] = shift_hist[0,1]

                #Apply Shifts
                for k in range(params[4]):
                    frame = rawData[(i*params[5])-params[4]+k]
                    row_shift = shifts[(i*params[5])-params[4]+k,0]
                    col_shift = shifts[(i*params[5])-params[4]+k,1]
                    out = apply_shift(frame, row_shift, col_shift, Nr, Nc, n, m)
                    registered[(i*params[5])-params[4]+k,:,:] = out


        row, col = fourier_imgreg(temp, int_frame, usfac=usfac, maxoff=maxoff)

        shift_hist = np.roll(shift_hist,1,axis=0)
        shift_hist[0,:] = cp.asnumpy(cp.array([row,col]))


    return registered, shifts, temps


def register_fastest(rawData, params, temps=[]):

    usfac = 100
    maxoff = 100

    t,n,m = rawData.shape   

    Nr = cp.fft.ifftshift(cp.linspace(-cp.fix(n / 2), cp.ceil(n / 2) - 1, n))
    Nc = cp.fft.ifftshift(cp.linspace(-cp.fix(m / 2), cp.ceil(m / 2) - 1, m))
    [Nc, Nr] = cp.meshgrid(Nc, Nr)

    # Define Template
    w = cp.ones((3,1,1))/3
    temp = cp.copy(rawData[0:params[5],params[0]:params[1],params[2]:params[3]])
    temp = temp/cp.amax(temp,axis=(1,2))[:,cp.newaxis,cp.newaxis]
    temp = cp.power(temp,params[6])
    temp = ndimage.convolve(temp,w)

    tnum = 0
    temp = cp.average(temp, axis=0)
    if len(temps) == 0:
        temps = [cp.copy(temp)]
    else:
        temp = cp.median(temps,axis=0) 

    # Preprocess Template
    temp = periodic_smooth_decomp(temp)[0]
    temp = temp-cp.average(temp)
    temp = cp.fft.fft2(temp)

    # Initialize Arrays
    registered = cp.zeros(rawData.shape)
    shifts = np.zeros((t,2))
    shift_hist = np.zeros((params[4]//params[5],2))
    shift_idx = np.arange(0,params[4])
    calc_shift_idx = np.arange(params[5]//2,params[4]+(params[5]//2),params[5])

    idx = t//params[5]
    p0 = np.array([5,0.1,0])

    for i in range(idx):

        int_frame = cp.copy(rawData[i*params[5]:(i+1)*params[5],params[0]:params[1],params[2]:params[3]])
        int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
        int_frame = cp.power(int_frame,params[6])
        #int_frame = ndimage.convolve(int_frame,w)
        int_frame = cp.average(int_frame,axis=0)

        if i > 0:
            if i % (params[4]//params[5]) == 0:
                # Calculate Interpolation Endpoints
                # Beginning
                int_frame = cp.copy(rawData[(i*params[5])-params[4]:(i*params[5])-params[4]+4,params[0]:params[1],params[2]:params[3]])
                int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                int_frame = cp.power(int_frame,params[6])
                int_frame = cp.average(int_frame,axis=0)
                first_row, first_col = fourier_imgreg(temp, int_frame, usfac=usfac, maxoff=maxoff)

                # End
                int_frame = cp.copy(rawData[(i*params[5])-4:(i*params[5]),params[0]:params[1],params[2]:params[3]])
                int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                int_frame = cp.power(int_frame,params[6])
                int_frame = cp.average(int_frame,axis=0)
                last_row, last_col = fourier_imgreg(temp, int_frame, usfac=usfac, maxoff=maxoff)

                # Initialize Interpolation
                shift_idxs = np.insert(calc_shift_idx,0,0)
                shift_idxs = np.append(shift_idxs,params[4])

                row_shifts = np.copy(shift_hist[:,0][::-1])
                row_shifts = np.insert(row_shifts,0,cp.asnumpy(first_row))
                row_shifts = np.append(row_shifts,cp.asnumpy(last_row))

                col_shifts = np.copy(shift_hist[:,1][::-1])
                col_shifts = np.insert(col_shifts,0,cp.asnumpy(first_col))
                col_shifts = np.append(col_shifts,cp.asnumpy(last_col))

                # Interpolate Shifts
                col_param, col_param_cov = curve_fit(sin_fit, shift_idxs, col_shifts, p0=p0)
                row_param, row_param_cov = curve_fit(sin_fit, shift_idxs, row_shifts, p0=p0)

                local_row_shifts = sin_fit(shift_idx, row_param[0], row_param[1], row_param[2])
                local_col_shifts = sin_fit(shift_idx, col_param[0], col_param[1], col_param[2])

                #
                # cs_row = interp1d(shift_idxs, row_shifts, kind='cubic')
                # cs_col = interp1d(shift_idxs, col_shifts, kind='cubic')

                # local_row_shifts = cs_row(shift_idx)
                # local_col_shifts = cs_col(shift_idx)

                shifts[(i*params[5])-params[4]:(i*params[5]),0] = local_row_shifts
                shifts[(i*params[5])-params[4]:(i*params[5]),1] = local_col_shifts

                #shifts[(i*params[5])-params[4]:(i*params[5])+(params[5]//2)-params[4],1] = shift_hist[-1,1]

                #shifts[(i*params[5])-params[4]+(params[5]//2):(i*params[5])-(params[5]//2),0] = local_row_shifts
                #shifts[(i*params[5])-params[4]+(params[5]//2):(i*params[5])-(params[5]//2),1] = local_col_shifts

                #shifts[(i*params[5])-(params[5]//2):(i*params[5]),0] = shift_hist[0,0]
                #shifts[(i*params[5])-(params[5]//2):(i*params[5]),1] = shift_hist[0,1]

                #Apply Shifts
                for k in range(params[4]):
                    frame = rawData[(i*params[5])-params[4]+k]
                    row_shift = shifts[(i*params[5])-params[4]+k,0]
                    col_shift = shifts[(i*params[5])-params[4]+k,1]
                    out = apply_shift(frame, row_shift, col_shift, Nr, Nc, n, m)
                    registered[(i*params[5])-params[4]+k,:,:] = out
                
                # Update Template
                tnum = tnum+1
                new = cp.array(registered[(i*params[5])-params[4]+1:(i*params[5])-params[4]+params[5]+1,params[0]:params[1],params[2]:params[3]])
                new = new/cp.amax(new,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                new = cp.power(new,params[6])
                new = cp.average(new,axis=0)
                if len(temps) > 10:
                    temps[tnum] = new
                else:
                    temps = cp.append(temps,[new],axis=0)
                temp = cp.median(temps,axis=0)
                temp = periodic_smooth_decomp(temp)[0]
                temp = temp - cp.average(temp)
                
                temp = cp.fft.fft2(temp)
                if tnum == 9:
                    tnum = 0
            
            elif i == idx-1:
                i = i+1
                # Calculate Interpolation Endpoints
                # Beginning
                int_frame = cp.copy(rawData[(i*params[5])-params[4]:(i*params[5])-params[4]+4,params[0]:params[1],params[2]:params[3]])
                int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                int_frame = cp.power(int_frame,params[6])
                int_frame = cp.average(int_frame,axis=0)
                first_row, first_col = fourier_imgreg(temp, int_frame, usfac=usfac, maxoff=maxoff)

                # End
                int_frame = cp.copy(rawData[(i*params[5])-4:(i*params[5]),params[0]:params[1],params[2]:params[3]])
                int_frame = int_frame/cp.amax(int_frame,axis=(1,2))[:,cp.newaxis,cp.newaxis]
                int_frame = cp.power(int_frame,params[6])
                int_frame = cp.average(int_frame,axis=0)
                last_row, last_col = fourier_imgreg(temp, int_frame, usfac=usfac, maxoff=maxoff)

                # Initialize Interpolation
                shift_idxs = np.insert(calc_shift_idx,0,0)
                shift_idxs = np.append(shift_idxs,params[4])

                row_shifts = np.copy(shift_hist[:,0][::-1])
                row_shifts = np.insert(row_shifts,0,cp.asnumpy(first_row))
                row_shifts = np.append(row_shifts,cp.asnumpy(last_row))

                col_shifts = np.copy(shift_hist[:,1][::-1])
                col_shifts = np.insert(col_shifts,0,cp.asnumpy(first_col))
                col_shifts = np.append(col_shifts,cp.asnumpy(last_col))

                # Interpolate Shifts
                col_param, col_param_cov = curve_fit(sin_fit, shift_idxs, col_shifts, p0=p0)
                row_param, row_param_cov = curve_fit(sin_fit, shift_idxs, row_shifts, p0=p0)

                local_row_shifts = sin_fit(shift_idx, row_param[0], row_param[1], row_param[2])
                local_col_shifts = sin_fit(shift_idx, col_param[0], col_param[1], col_param[2])

                # local_row_shifts = sin_fit(shift_idx, row_param[0], row_param[1], row_param[2])
                # local_col_shifts = sin_fit(shift_idx, col_param[0], col_param[1], col_param[2])

                # cs_row = interp1d(shift_idxs, row_shifts, kind='cubic')
                # cs_col = interp1d(shift_idxs, col_shifts, kind='cubic')

                # local_row_shifts = cs_row(shift_idx)
                # local_col_shifts = cs_col(shift_idx)

                shifts[(i*params[5])-params[4]:(i*params[5]),0] = local_row_shifts
                shifts[(i*params[5])-params[4]:(i*params[5]),1] = local_col_shifts

                #shifts[(i*params[5])-params[4]:(i*params[5])+(params[5]//2)-params[4],1] = shift_hist[-1,1]

                #shifts[(i*params[5])-params[4]+(params[5]//2):(i*params[5])-(params[5]//2),0] = local_row_shifts
                #shifts[(i*params[5])-params[4]+(params[5]//2):(i*params[5])-(params[5]//2),1] = local_col_shifts

                #shifts[(i*params[5])-(params[5]//2):(i*params[5]),0] = shift_hist[0,0]
                #shifts[(i*params[5])-(params[5]//2):(i*params[5]),1] = shift_hist[0,1]

                #Apply Shifts
                for k in range(params[4]):
                    frame = rawData[(i*params[5])-params[4]+k]
                    row_shift = shifts[(i*params[5])-params[4]+k,0]
                    col_shift = shifts[(i*params[5])-params[4]+k,1]
                    out = apply_shift(frame, row_shift, col_shift, Nr, Nc, n, m)
                    registered[(i*params[5])-params[4]+k,:,:] = out


        row, col = fourier_imgreg(temp, int_frame, usfac=usfac, maxoff=maxoff)

        shift_hist = np.roll(shift_hist,1,axis=0)
        shift_hist[0,:] = cp.asnumpy(cp.array([row,col]))


    return registered, shifts, temps


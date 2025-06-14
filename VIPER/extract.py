import numpy as np
from scipy import ndimage, signal, sparse
import zarr
from tqdm import tqdm
import dask
import dask.array as da
import sys
import os
import scipy.io as sio
import tifffile as tif
from datetime import datetime


def context_region(clnmsk, pix_pad=0):
    n,m = clnmsk.shape
    rows = np.any(clnmsk, axis=1)
    cols = np.any(clnmsk, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    top = rmin-pix_pad
    if top < 0:
        top = 0
    bot = rmax+pix_pad
    if bot > n:
        bot = n
    lef = cmin-pix_pad
    if lef < 0:
        lef = 0
    rig = cmax+pix_pad
    if rig > m:
        rig = m
    
    return top,bot,lef,rig

def pb_correct(im_data,fs):
    medn = np.median(im_data,axis=0)
    im_data = (im_data - medn[np.newaxis,:])
    bfilt,afilt = signal.butter(2, 1/3,'highpass', fs=fs)
    clean_data = signal.filtfilt(bfilt,afilt,im_data, axis=0)    
    return clean_data

def define_background(imageData, box):
    # Define Background
    n,m = imageData[1].shape
    t1 = box[0]-35
    t2 = box[0]-40
    b1 = box[1]+35
    b2 = box[1]+40
    l1 = box[2]-35
    l2 = box[2]-40
    r1 = box[3]+35
    r2 = box[3]+40
    
    if t1 < 0:
        t1 = 0
    if t2 < 0:
        t2 = 0
    if b1 > n:
        b1 = n
    if b2 > n:
        b2 = n
    if l1 < 0 :
        l1 = 0
    if l2 < 0:
        l2 = 0
    if r1 > m:
        r1 = m
    if r2 > m:
        r2 = m
        
    bg_top = imageData[:,t2:t1,l2:r2]
    bg_bot = imageData[:,b1:b2,l2:r2]
    bg_lef = imageData[:,t1:b1,l2:l1]
    bg_rig = imageData[:,t1:b1,r1:r2]
    
    bg_pix = np.concatenate((bg_top.reshape(len(imageData),-1),bg_bot.reshape(len(imageData),-1),bg_lef.reshape(len(imageData),-1),bg_rig.reshape(len(imageData),-1)),axis=1)
    
    if b1 == b2:
        bg_pix = np.concatenate((bg_top.reshape(len(imageData),-1),bg_lef.reshape(len(imageData),-1),bg_rig.reshape(len(imageData),-1)),axis=1)
    
    if t1 == t2:
        bg_pix = np.concatenate((bg_bot.reshape(len(imageData),-1),bg_lef.reshape(len(imageData),-1),bg_rig.reshape(len(imageData),-1)),axis=1)
        
    if l1 == l2:
        bg_pix = np.concatenate((bg_top.reshape(len(imageData),-1),bg_bot.reshape(len(imageData),-1),bg_rig.reshape(len(imageData),-1)),axis=1)
    
    if r1 == r2:
        bg_pix = np.concatenate((bg_top.reshape(len(imageData),-1),bg_bot.reshape(len(imageData),-1),bg_lef.reshape(len(imageData),-1)),axis=1)
        
    #bg_pix = cp.array(bg_pix)
    
    return bg_pix

def background_svd(background_px, npc=4, lmd=0.01):
    background_px = da.from_array(background_px)
    print(background_px)
    #U,Z,V = np.linalg.svd(background_px)
    U, Z, V = da.linalg.svd(background_px)
    U.compute()
    U = np.array(U)
    Ub = U[:,0:npc]
    UbT = np.transpose(Ub)
    a = np.matmul(UbT,Ub)
    fnorm = np.sum(np.square(Ub))
    I = np.identity(npc)
    b = np.linalg.inv(a+lmd*fnorm*I)
    
    return Ub, UbT, b

# def initial_extraction(imageData, ROIs, pole, fs=500):

#     traces = np.zeros((len(ROIs),len(imageData)))
#     masks = np.asarray([object]*len(ROIs), dtype=object)

#     for i in range(len(ROIs)):
#         mask = ROIs[i]
#         row_min, row_max, col_min, col_max = context_region(mask,pix_pad=3)
        
#         mask = mask[row_min:row_max, col_min:col_max]
#         context = imageData[:,row_min:row_max, col_min:col_max]

#         if pole:
#             context = -context

#         context = pb_correct(context.reshape(len(context),-1),fs=fs)
#         context = context.reshape(len(context),row_max-row_min,col_max-col_min)
        
#         trace = np.average(context*mask[np.newaxis,:,:],axis=(1,2))
#         traces[i] = trace
#         masks[i] = mask  

#     results = {
#         'Masks': masks,
#         'DFF': traces,
#         'SpikeTemplate': np.zeros((len(ROIs),11)),
#         'SpikeSNR': np.zeros(len(ROIs))
#         }
    
#     return results


# def extract_source_trace(imageData, ROI, pole, fs=500, iters=3, thresh=6):
#     box = context_region(ROI)
#     mask = ROI[box[0]:box[1], box[2]:box[3]]
#     context = imageData[:,box[0]:box[1], box[2]:box[3]]
#     background = define_background(imageData,box)


#     if pole:
#         context = -context
#         background = -background
        
#     context = pb_correct(context.reshape(len(context),-1),fs=fs)
#     context = context.reshape(len(context),box[1]-box[0],box[3]-box[2])
#     background = pb_correct(background,fs=fs)
    
#     # Initial Spike Detection Filter
#     bfilt,afilt = signal.butter(3,3,'highpass',fs=fs)
#     Ub, UbT, b = background_svd(background)

#     for j in range(iters+1):
#         # Extract first trace
#         trace = context*mask[np.newaxis,:,:]
#         trace = np.average(trace,axis=(1,2))
        
#         # Background Subtraction
#         beta = np.matmul(np.matmul(b,UbT),trace)
#         trace = trace - np.matmul(Ub,beta)
        
#         # Initial Spike Detection
#         extract_spk_1 = np.copy(trace) - np.median(trace)
#         extract_spk_1 = signal.filtfilt(bfilt,afilt,extract_spk_1)
#         th = np.std(-extract_spk_1[extract_spk_1 < 0])*thresh
#         spikeIdx = signal.find_peaks(extract_spk_1,height=th)[0]
        
#         # Create Initial Spike Template
#         spk_win = 5
#         spikes = spikeIdx[(spikeIdx > 10) & (spikeIdx < len(extract_spk_1)-10)]
#         spk_mat = np.zeros((len(spikes),(2*spk_win)+1))
        
#         for k in range(len(spikes)):
#             spk = np.copy(trace[spikes[k]-spk_win:spikes[k]+spk_win+1])
#             spk_mat[k] = spk - np.amin(spk)
            
#         spk_template = np.average(spk_mat,axis=0)
        
#         spiketrain = np.zeros(len(trace))
#         spiketrain[spikeIdx] = 1
#         qq = signal.convolve(spiketrain, np.ones((2*spk_win)+1),'same')
#         noise = extract_spk_1[qq<0.5]
        
#         # Signal pre-whiten from welch spectral analysis
#         freq,pxx = signal.welch(noise,nfft=len(extract_spk_1))
#         pxx = np.sqrt(pxx)
        
#         if len(trace)%2 == 0:
#             pxx = np.append(pxx[:-1],np.flip(pxx[:-1]))
#         else:
#             pxx = np.append(pxx,np.flip(pxx[:-1])) 
        
#         # Pre-whiten signal
#         extract_spk_1 = np.fft.fft(np.copy(extract_spk_1))/pxx
#         extract_spk_1 = np.real(np.fft.ifft(extract_spk_1))
        
#         spk_mat = np.zeros((len(spikes),(2*spk_win)+1))
        
#         for k in range(len(spikes)):
#             spk = np.copy(extract_spk_1[spikes[k]-spk_win:spikes[k]+spk_win+1])
#             spk_mat[k] = spk - np.amin(spk)
            
#         # Match filter spike extraction from whitened signal
#         match_temp = np.average(spk_mat,axis=0)
#         extract_spk_1 = signal.correlate(extract_spk_1,match_temp,'same') 
#         th = np.std(-extract_spk_1[extract_spk_1 < 0])*(thresh*(3/4))
#         spikeIdx = signal.find_peaks(extract_spk_1,height=th)[0]
        
#         if len(spikeIdx) < 1:
#             snr = 0
#             print("No Spikes Detected")
#             break
        
#         if j == range(iters+1)[-1]:    
#             spiketrain = np.zeros(len(trace))
#             spiketrain[spikeIdx] = 1
#             qq = signal.convolve(spiketrain, np.ones((2*spk_win)+1),'same')
#             sign = np.amax(spk_template) - np.amin(spk_template)
#             nois = np.std(trace[qq<0.5])
#             snr = sign/nois
#             break
        
#         # Reconstruct trace from spike template
#         spiketrain = np.zeros(len(trace))
#         spiketrain[spikeIdx] = 1
#         trec = signal.convolve(spiketrain,spk_template, 'same')
        
#         # Refine Spatial Footprint
#         context = context.reshape(len(context),-1)
#         solver_params = sparse.linalg.lsmr(context,trec, damp=0.01, maxiter=1)
#         mask = solver_params[0]
#         mask[mask < np.average(mask)/2] = 0
#         mask = mask/np.amax(mask)
#         mask = np.reshape(mask,(box[1]-box[0],box[3]-box[2]))
#         context = np.reshape(context,(len(context),box[1]-box[0],box[3]-box[2]))
    
#     return mask, trace, spikeIdx, spk_template, snr



def trace_extraction(imageData, ROIs, pole, fs, iters=3, thresh=6):
    
    # Initialize Outputs
    final_masks = np.asarray([object]*len(ROIs), dtype=object)
    final_traces = np.asarray([object]*len(ROIs), dtype=object)
    final_spikes = np.asarray([object]*len(ROIs), dtype=object)
    final_spike_temps = np.asarray([object]*len(ROIs), dtype=object)
    final_snrs = np.zeros(len(ROIs))
    
    for i in range(len(ROIs)):
        # Focus on Region of Interest
        mask = ROIs[i]
        row_min, row_max, col_min, col_max = context_region(mask,pix_pad=5)
        
        mask = mask[row_min:row_max, col_min:col_max]
        context = imageData[:,row_min:row_max, col_min:col_max]
        background = define_background(imageData,[row_min, row_max, col_min, col_max])
        print(context.shape)
        print(background.shape)

        if pole:
            context = -context
            background = -background
        
        context = pb_correct(context.reshape(len(context),-1),fs=fs)
        context = context.reshape(len(context),row_max-row_min,col_max-col_min)
        background = pb_correct(background,fs=fs)
        
        # Initial Spike Detection Filter
        bfilt,afilt = signal.butter(3,3,'highpass',fs=fs)
        Ub, UbT, b = background_svd(background)
        print('background')

        
        for j in range(iters+1):
            print('iter '+str(j))
            print(context.shape)
            print(mask.shape)
            # Extract first trace
            trace = context*mask[np.newaxis,:,:]
            trace = np.average(trace,axis=(1,2))
            
            # Background Subtraction
            beta = np.matmul(np.matmul(b,UbT),trace)
            trace = trace - np.matmul(Ub,beta)
            
            # Initial Spike Detection
            extract_spk_1 = np.copy(trace) - np.median(trace)
            extract_spk_1 = signal.filtfilt(bfilt,afilt,extract_spk_1)
            th = np.std(-extract_spk_1[extract_spk_1 < 0])*thresh
            spikeIdx = signal.find_peaks(extract_spk_1,height=th)[0]
            
            # Create Initial Spike Template
            spk_win = 5
            spikes = spikeIdx[(spikeIdx > 10) & (spikeIdx < len(extract_spk_1)-10)]
            spk_mat = np.zeros((len(spikes),(2*spk_win)+1))
            
            for k in range(len(spikes)):
                spk = np.copy(trace[spikes[k]-spk_win:spikes[k]+spk_win+1])
                spk_mat[k] = spk - np.amin(spk)
                
            spk_template = np.average(spk_mat,axis=0)
            
            spiketrain = np.zeros(len(trace))
            spiketrain[spikeIdx] = 1
            qq = signal.convolve(spiketrain, np.ones((2*spk_win)+1),'same')
            noise = extract_spk_1[qq<0.5]
            
            # Signal pre-whiten from welch spectral analysis
            freq,pxx = signal.welch(noise,nfft=len(extract_spk_1))
            pxx = np.sqrt(pxx)
            
            if len(trace)%2 == 0:
                pxx = np.append(pxx[:-1],np.flip(pxx[:-1]))
            else:
                pxx = np.append(pxx,np.flip(pxx[:-1])) 
            
            # Pre-whiten signal
            extract_spk_1 = np.fft.fft(np.copy(extract_spk_1))/pxx
            extract_spk_1 = np.real(np.fft.ifft(extract_spk_1))
            
            spk_mat = np.zeros((len(spikes),(2*spk_win)+1))
            
            for k in range(len(spikes)):
                spk = np.copy(extract_spk_1[spikes[k]-spk_win:spikes[k]+spk_win+1])
                spk_mat[k] = spk - np.amin(spk)
                
            # Match filter spike extraction from whitened signal
            match_temp = np.average(spk_mat,axis=0)
            extract_spk_1 = signal.correlate(extract_spk_1,match_temp,'same') 
            th = np.std(-extract_spk_1[extract_spk_1 < 0])*(thresh*(3/4))
            spikeIdx = signal.find_peaks(extract_spk_1,height=th)[0]
            
            if len(spikeIdx) < 1:
                snr = 0
                print("No Spikes Detected")
                break
            
            if j == range(iters+1)[-1]:    
                spiketrain = np.zeros(len(trace))
                spiketrain[spikeIdx] = 1
                qq = signal.convolve(spiketrain, np.ones((2*spk_win)+1),'same')
                sign = np.amax(spk_template) - np.amin(spk_template)
                nois = np.std(trace[qq<0.5])
                snr = sign/nois
                break
            
            # Reconstruct trace from spike template
            spiketrain = np.zeros(len(trace))
            spiketrain[spikeIdx] = 1
            trec = signal.convolve(spiketrain,spk_template, 'same')
            
            # Refine Spatial Footprint
            context = context.reshape(len(context),-1)
            solver_params = sparse.linalg.lsmr(context,trec, damp=0.01, maxiter=1)
            mask = solver_params[0]
            mask[mask < np.average(mask)/2] = 0
            mask = mask/np.amax(mask)
            mask = np.reshape(mask,(row_max-row_min,col_max-col_min))
            context = np.reshape(context,(len(context),row_max-row_min,col_max-col_min))
            
        final_masks[i] = mask
        final_traces[i] = trace
        final_spikes[i] = spikeIdx
        final_spike_temps[i] = spk_template
        final_snrs[i] = snr
    
    results = {
        'Masks': final_masks,
        'DFF': final_traces,
        'Spikes': final_spikes,
        'SpikeTemplate': final_spike_temps,
        'SpikeSNR': final_snrs
        }
            
    return results
            
            
            
if __name__ == '__main__':
    reg_data_path = str(sys.argv[1])
    roi_data_path = str(sys.argv[2])
    pole = bool(int(sys.argv[3]))
    fs = int(sys.argv[4])
    output_data_path = str(sys.argv[5])

    registered_data = zarr.open(reg_data_path)
    rois = tif.imread(roi_data_path)

    results = trace_extraction(registered_data, rois, pole, fs)

    now = datetime.now()
    saved_dir = os.path.join(output_data_path, 'saved')
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    date_time = now.strftime("%Y-%m-%d_%H%M%S")
    output_file_name = "VIP_Saved_" + date_time
    np.save(os.path.join(saved_dir, output_file_name+'.npy'), results)
    sio.savemat(os.path.join(saved_dir, output_file_name+'.mat'), results)



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    

    

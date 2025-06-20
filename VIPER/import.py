import cv2
import zarr
import numpy as np
import os
import sys
from scipy import signal
try:
    import cupy as cp
    from cupyx.scipy import signal as csignal
except:
    import numpy as cp
    import scipy.signal as csignal
    pass
import tifffile as tif
import glob
import multiprocessing
import sys

def downsample2d(inputArray, kernelSize, gpu_mode=False):

    """
    Downsample 2D IMage
    
    Parameters
    ----------
    inputArray : Input Image
    kernelSize : Downsampling factor

    Returns
    -------
    downsampled_array
    """

    n,m = inputArray.shape
    data_type = inputArray.dtype
    row_crop = n%kernelSize
    col_crop = m%kernelSize
    if gpu_mode:
        average_kernel = (1/(kernelSize*kernelSize))*cp.ones((kernelSize,kernelSize),dtype=data_type)
        blurred_array = csignal.convolve2d(inputArray, average_kernel, mode='same')
    else:
        average_kernel = (1/(kernelSize*kernelSize))*np.ones((kernelSize,kernelSize),dtype=data_type)
        blurred_array = signal.convolve2d(inputArray, average_kernel, mode='same')

    
    if row_crop > 0:
        blurred_array = blurred_array[row_crop:,:]
    else:
        pass
    if col_crop > 0:
        blurred_array = blurred_array[:,col_crop:]
    else:
        pass
    
    downsampled_array = blurred_array[::kernelSize,::kernelSize]

    return downsampled_array


def downsample3d(inputArray, kernelSize, gpu_mode=False):
    """
    Downsample Video (3D Array)
    
    Parameters
    ----------
    inputArray : Input Array (3D)
    kernelSize : Downsampling Factor (for each 2D image)

    Returns
    -------
    downsampled array

    """
    first_downsampled = downsample2d(inputArray[0,:,:], kernelSize, gpu_mode=gpu_mode)
    data_type = first_downsampled.dtype
    if gpu_mode:
        downsampled_array = cp.zeros((inputArray.shape[0],first_downsampled.shape[0], first_downsampled.shape[1]),dtpye = data_type)
    else:
        downsampled_array = np.zeros((inputArray.shape[0],first_downsampled.shape[0], first_downsampled.shape[1]),dtpye = data_type)

    downsampled_array[0,:,:] = first_downsampled

    for i in range(1, inputArray.shape[0]):
        downsampled_array[i,:,:] = downsample2d(inputArray[i,:,:], kernelSize, gpu_mode=gpu_mode)

    return downsampled_array



def load_avi(fname, ds, tempFolder, gpu_mode=False):
    '''
    Load .avi data and create a memory mapped file

    Returns
    -------
    imageData : mapped raw Data
    fname : full file name
    ds : downsampled factor

    '''

    #byte_size = 50e6  # Mapped file chunk size in bytes (50 MB)

    cap = cv2.VideoCapture(fname)
    success, frame = cap.read()
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Initialize video dimensions
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n,m = gray.shape
    data_type = frame.dtype
    #frames_per_chunk = int(byte_size/(frame.size*frame.itemsize))
    
    if ds > 1:
        n = int(n/ds)
        m = int(m/ds)
        gray = downsample2d(gray, ds)
        
    mapped_file_path = os.path.normpath(tempFolder + '/raw_map.zarr')
    imageData = zarr.open(mapped_file_path, mode='a', shape=(length,n,m), chunks=(1,n,m), dtype = data_type) # Create memory mapped file
    imageData[0, :, :] = gray
    # Load video frames into mapped file

    #progress_bar = iter(tqdm(range(length), desc='Mapping Data: '))
    for i in range(length-1):
        success, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ds > 1:
            gray = downsample2d(gray, ds)
        imageData[i+1, :, :] = gray

       # next(progress_bar)
    
    cap.release()
    
    return imageData

def map_frames(imageData, start_idx, files, fname, ds, gpu_mode):
    
    for i, file in enumerate(files):
        image = tif.imread(os.path.join(fname, file))
        if ds > 1:
            if gpu_mode:
                image = downsample2d(cp.array(image), ds, gpu_mode=gpu_mode)
                imageData[start_idx+i, :, :] = cp.asnumpy(image)
            else:
                image = downsample2d(image, ds, gpu_mode=gpu_mode)
                imageData[start_idx+i, :, :] = image
        else:
            imageData[start_idx+i, :, :] = image



if __name__ == "__main__":

    importPath = str(sys.argv[1])
    tempPath = str(sys.argv[2])
    ds = int(sys.argv[3])
    gpu_mode = bool(int(sys.argv[4]))

    # Check for different file formats and load the data
    dirs = importPath.split("/")
    dirs = dirs[-1]

    # Load data from .tiff stack
    if (glob.glob(importPath+'/*.tiff')) or (glob.glob(importPath+'/*.tif')):
        #byte_size = 50e6 # Mapped file chunk size in bytes (50 MB)

        data_path = glob.glob(importPath+'/*.tif')[0]
        store = tif.imread(data_path, aszarr=True)
        rawData = zarr.open(store, mode='r')
        t,n,m = rawData.shape
        data_type = rawData.dtype
        frame = rawData[0]
        #frames_per_chunk = int(byte_size/(frame.size*frame.itemsize))
        mapped_file_path = tempPath + '/raw_map.zarr'

        if ds > 1:
            n = int(n/ds)
            m = int(m/ds)
            imageData = zarr.open(mapped_file_path, mode='a', shape=(t,n,m), chunks=(1,n,m), dtype = data_type)
            for i in range(t):
                if gpu_mode:
                    imageData[:,] = cp.asnumpy(downsample2d(cp.array(rawData[i,:,:]), ds, gpu_mode=gpu_mode))
                else:
                    imageData[:,] = downsample2d(rawData[i,:,:], ds, gpu_mode=gpu_mode)
        else:
            imageData = rawData
            
    # Load data from .avi file          
    if glob.glob(importPath+'/*.avi'):
        data_path = glob.glob(importPath+'/*.avi')[0]
        imageData = load_avi(data_path,ds,tempPath, gpu_mode=gpu_mode)

    # Load data from .tif directory
    if glob.glob(importPath+'/'+dirs):
        fname = os.path.normpath(importPath+'/'+dirs)
        files = [f for f in os.listdir(fname) if f.endswith('.tif')]
        files.sort(key=lambda x: '{0:0>15}'.format(x).lower())
        length = len(files)
        cores = multiprocessing.cpu_count()

        if cores > 12:
            cores = 12

        files_per_core = int(length // cores)
        frame = tif.imread(os.path.join(fname, files[0]))
        n, m = frame.shape
        data_type = frame.dtype

        if ds > 1:
            n = int(n / ds)
            m = int(m / ds)

        mapped_file_path = os.path.normpath(tempPath + '/raw_map.zarr')
        #synchronizer = os.path.normpath(tempPath + '/raw_map.sync')
        #synchronizer = zarr.ProcessSynchronizer(synchronizer)
        #imageData = zarr.open_array(mapped_file_path, mode='a', shape=(length, n, m), chunks=(1, n, m), dtype=data_type, synchronizer=synchronizer)
        imageData = zarr.open_array(mapped_file_path, mode='a', shape=(length, n, m), chunks=(1, n, m), dtype=data_type)


        # Load frames into mapped file using multiprocessing
        procs = np.empty(cores,dtype=object)
        
        for i in range(cores):

            if i < cores-1:
                procs[i] = multiprocessing.Process(target=map_frames,args=(imageData,files_per_core*i,files[files_per_core*i:files_per_core*(i+1)], fname, ds, gpu_mode))

            else:
                procs[i] = multiprocessing.Process(target=map_frames,args=(imageData,files_per_core*i,files[files_per_core*i:], fname, ds, gpu_mode))

            procs[i].start()


        for i in range(cores):
            procs[i].join()



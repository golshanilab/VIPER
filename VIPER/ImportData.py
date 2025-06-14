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
import subprocess
import traceback

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



def load_avi(fname, ds, gpu_mode=False):
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
        
    mapped_file_path = fname.split('.')[0] + '_raw_map.zarr'
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



def import_data(directory,temp, ds=1, GPU_mode=False):

    # Check for different file formats and load the data
    dirs = directory.split("/")
    dirs = dirs[-1]

    # Load data from .tiff stack
    if (glob.glob(directory+'/*.tiff')) or (glob.glob(directory+'/*.tif')):
        #byte_size = 50e6 # Mapped file chunk size in bytes (50 MB)

        data_path = glob.glob(directory+'/*.tif')[0]
        store = tif.imread(data_path, aszarr=True)
        rawData = zarr.open(store, mode='r')
        t,n,m = rawData.shape
        data_type = rawData.dtype
        frame = rawData[0]
        #frames_per_chunk = int(byte_size/(frame.size*frame.itemsize))
        mapped_file_path = directory + '/raw_map.zarr'

        if ds > 1:
            n = int(n/ds)
            m = int(m/ds)
            imageData = zarr.open(mapped_file_path, mode='a', shape=(t,n,m), chunks=(1,n,m), dtype = data_type)
            for i in range(t):
                if GPU_mode:
                    imageData[:,] = cp.asnumpy(downsample2d(cp.array(rawData[i,:,:]), ds, gpu_mode=GPU_mode))
                else:
                    imageData[:,] = downsample2d(rawData[i,:,:], ds, gpu_mode=GPU_mode)
        else:
            imageData = rawData
            
    # Load data from .avi file          
    if glob.glob(directory+'/*.avi'):
        data_path = glob.glob(directory+'/*.avi')[0]
        imageData = load_avi(data_path,ds, gpu_mode=GPU_mode)

    # Load data from .tif directory
    if glob.glob(directory+'/'+dirs):
        data_path = os.path.normpath(glob.glob(directory+'/'+dirs)[0])
        directory = os.path.normpath(directory)
        script_path = os.path.normpath(os.getcwd()+'/VIPER/import_tif_dir.py')
        #command_string = activate_path+' python '+script_path+' '+data_path+' '+directory+' '+str(ds)+' '+str(GPU_mode)
        try:
            p = subprocess.Popen([sys.executable, str(script_path), str(data_path), str(directory), str(ds), str(GPU_mode), str(temp)])
            p.wait()
        except Exception as e:
            print(e)
            print(traceback.format_exc())

        imageData = os.path.normpath(temp+'/raw_map.zarr')
        imageData = zarr.open(imageData, mode='r')

    chunks = chunk_data(imageData)
        
    data_and_chunks = {
        'MappedData':imageData,
        'MaxChunkMem':chunks[2],
        'ChunkNum':chunks[0],
        'FramesPerChunk': chunks[1],
        }
        
    return data_and_chunks

def chunk_data(imageData):
    
    max_memory = int(2e9)  # 2GB
    
    t,m,n = imageData.shape
    frame = imageData[0]
    frame_data_size = frame.size*frame.itemsize #frame in bytes
    
    frames_per_chunk = int(max_memory//frame_data_size)
    num_chunks = int(t//frames_per_chunk)
    
    if num_chunks < t/frames_per_chunk:
        num_chunks = int(num_chunks+1)
    
    
    return np.array([num_chunks,frames_per_chunk,max_memory])




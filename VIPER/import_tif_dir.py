import tifffile as tif
import os
import numpy as np
import zarr
import sys
import multiprocessing
import time
try:
    import cupy as cp
except:
    import numpy as cp
    pass
from ImportData import downsample2d


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


if __name__ == '__main__':

    fname = os.path.normpath(str(sys.argv[1]))
    directory = os.path.normpath(str(sys.argv[2]))
    ds = int(sys.argv[3])
    gpu_mode = bool(sys.argv[4])
    output_directory = os.path.normpath(str(sys.argv[5]))


    files = [f for f in os.listdir(fname) if f.endswith('.tif')]
    files.sort(key=lambda x: '{0:0>15}'.format(x).lower())
    length = len(files)
    cores = multiprocessing.cpu_count()

    if cores> 12:
        cores = 12

    files_per_core = int(length//cores)
    frame = tif.imread(os.path.join(fname, files[0]))
    n, m = frame.shape
    data_type = frame.dtype

    if ds > 1:
        n = int(n/ds)
        m = int(m/ds)
    

    mapped_file_path = os.path.normpath(output_directory+'/raw_map.zarr')
    synchronizer = os.path.normpath(output_directory+'/raw_map.sync')
    synchronizer = zarr.ProcessSynchronizer(synchronizer)
    imageData = zarr.open_array(mapped_file_path, mode='a', shape=(length, n, m), chunks =(1,n,m), dtype=data_type, synchronizer=synchronizer)

    # Load frames into mapped file
    print('Loading Frames: ')
    
    procs = np.empty(cores,dtype=object)

    for i in range(cores):

        if i < cores-1:
            procs[i] = multiprocessing.Process(target=map_frames,args=(imageData,files_per_core*i,files[files_per_core*i:files_per_core*(i+1)], fname, ds, gpu_mode))

        else:
            procs[i] = multiprocessing.Process(target=map_frames,args=(imageData,files_per_core*i,files[files_per_core*i:], fname, ds, gpu_mode))

        procs[i].start()


    for i in range(cores):
        procs[i].join()




    

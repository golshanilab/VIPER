import zarr
import numpy as np
import sys
import os
import multiprocessing

def rechunk_data(input_file, output_file, start_idx, frames):

    for i in range(frames):
        image = input_file[start_idx+i, :, :]
        output_file[start_idx+i, :, :] = image

if __name__ == '__main__':
    input_path = os.path.normpath(str(sys.argv[1]))
    output_path = os.path.normpath(str(sys.argv[2]))
    cores = multiprocessing.cpu_count()

    input_file = zarr.open(input_path, mode='a')
    temp = input_file[0]
    data_type = temp.dtype
    n, m = temp.shape
    t = input_file.shape[0]
    frames = int(t//cores)

    sync = os.path.normpath(output_path+'/remapped_data.sync')
    sync = zarr.ProcessSynchronizer(sync)
    output_file = zarr.open(output_path+'/remapped_data.zarr', mode='a', shape=(t, n, m), chunks=(t, 1, 1), dtype=data_type, synchronizer=sync)

    if cores > 12:
        cores = 12

    procs = np.empty(cores, dtype=object)

    for i in range(cores):
        
        if i < cores-1:
            procs[i] = multiprocessing.Process(target=rechunk_data, args=(input_file, output_file, frames*i, frames))
        else:
            procs[i] = multiprocessing.Process(target=rechunk_data, args=(input_file, output_file, frames*i, t-frames*i))
        procs[i].start()

    for  i in range(cores):
        procs[i].join()


    


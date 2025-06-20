import os
import zarr
import shutil
from rechunker import rechunk
import sys


def run(registeredDataPath):
    directory = os.path.dirname(registeredDataPath)
    intermed = os.path.join(directory,'intermediate_data.zarr')
    remapped = os.path.join(directory,'remapped_data.zarr')
    if os.path.exists(intermed):
        shutil.rmtree(intermed)
    if os.path.exists(remapped):
        shutil.rmtree(remapped)
    
    source = zarr.open(registeredDataPath)
    frame = source[0]
    t,n,m = source.shape

    remapping = rechunk(source, (t,1,1), "1GB", remapped, temp_store=intermed)
    remapping.execute()
    
if __name__ == '__main__':
    path = os.path.normpath(str(sys.argv[1]))
    run(path)
    
    
    
    
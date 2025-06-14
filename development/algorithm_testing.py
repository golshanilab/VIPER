import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import zarr
import tifffile as tif
from VIPER.ImportData import import_data
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
import os
from VIPER.Registration_gpu import register as register
import time


def set_up(fname):

    data = import_data(fname,GPU_mode=True)
    data = data['MappedData']
    t,n,m = data.shape
    data_type = data.dtype
    shifts = np.zeros((t,2))
    registeredData = zarr.open(os.path.join(fname, 'registered_data.zarr'), mode='a', shape=(t,n,m), chunks=(1,n,m), dtype=data_type)

    return data, shifts, registeredData


def test_registration(data, shifts, registeredData, params, fname):

    t,n,m = data.shape
    frame = data[0]
    full_data = (frame.itemsize)*n*m*t
    chunk_num = int(full_data//(1e9))
    mempool = cp.get_default_memory_pool()

    if chunk_num == 0:
        chunk_num = 1

    frames_per_chunk = int(t//chunk_num)
    temps = []

    t1 = np.round(time.time()*1000,2)
    for i in range(chunk_num):
        print(f"Processing chunk {i} out of {chunk_num}")
        if i == chunk_num-1:
            end = t
        else:
            end = (i+1)*frames_per_chunk
                
        start = i*frames_per_chunk
        chunk_to_register = data[start:end,:,:]

        print("Start: ", start)
        print("End: ", end)

        registered,shift,temps = register(cp.array(chunk_to_register), params, temps=temps)
        registeredData[start:end,:,:] = cp.asnumpy(registered)
        shifts[start:end,:] = shift
        del registered,shift
        mempool.free_all_blocks()
    
    t1 = np.round(time.time()*1000,2)-t1
    print(f"Time to register: {t1} ms")
    np.savetxt(os.path.join(fname, 'shifts.txt'), shifts, delimiter=',')
    return shifts, registeredData

def accuracy_measuerment(imageData):

    t = len(imageData)
    coefs = np.zeros(t)
    avg = np.average(imageData, axis=0)

    for i in range(t):
        cc = np.corrcoef(avg.flatten(),imageData[i].flatten())[0,1]
        coefs[i] = cc
    
    crisp = np.gradient(avg/np.amax(avg))
    crisp = np.linalg.norm(np.absolute(crisp))

    plt.plot(np.arange(0,t),coefs)
    plt.annotate(f"Crisp: {np.round(crisp,2)}", (t*(3/4),np.amax(coefs)))
    plt.xlabel('Frame')
    plt.ylabel('Correlation Coefficient')
    plt.title('10 Frame Average Linear Interpolation')
    plt.show()


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self, data, shfits):
        super().__init__()
        self.data = np.array(data)
        self.shifts = shifts

        self.setWindowTitle("My App")
        video = pg.ImageView()
        video.setImage(np.rot90(self.data,axes=(1,2)))

        # Set the central widget of the Window.
        self.setCentralWidget(video)


if __name__ == "__main__":

    fname = os.path.normpath('/home/administrator/Desktop/Data/Descartes/20240626/t1/')
    params = [50, 450, 100, 600, 200, 20, 4]
    data, shifts, registeredData = set_up(fname)
    shifts, registeredData = test_registration(data, shifts, registeredData, params, fname)

    #shifts = np.loadtxt('/home/administrator/Desktop/Data/Descartes/20240626/t1/shifts.txt', delimiter=',')
    #registeredData = zarr.open('/home/administrator/Desktop/Data/Descartes/20240626/t1/registered_data.zarr')

    app = QApplication(sys.argv)
    window = MainWindow(registeredData, shifts)
    window.show()
    app.exec()

    #accuracy_measuerment(registeredData)

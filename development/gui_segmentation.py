import sys
import os
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import tifffile as tif
import numpy as np
import zarr
import glob


#Creates the main application
class DataCuratorApp(QtWidgets.QWidget):


#python development/gui_segmentation.py


#source ~/Desktop/VIP/bin/activate



    

    def __init__(self):
        super().__init__()  # Initialize the QWidget
        self.data = []  # To store the loaded image data
        self.default_path = 'Registered.tif'  # Default folder path (update as needed)
        self.loadData(self.default_path)  # Load image data from the default path
        self.initUI()  # Initialize the user interface


       



        self.avg_frame = None
        self.dynamic_image = None

        self.levels_dict = {}
        self.gradient_dict = {}

        


        self.avg_levels = None
        self.dynamic_levels = None


        self.avg_gradient = None
        self.dynamic_gradient = None

    def loadData(self, path):
        self.data = []  # Reset the data list
        self.avg_frame = None
        self.dynamic_image = None

        #data_path = glob.glob(self.path+'/*.tif')[0]
        store = tif.imread(self.default_path, aszarr=True)
        self.data = zarr.open(store, mode='r')

        #for file_name in sorted(os.listdir(path)):  # Loop through sorted file names in the folder
        #    if file_name.endswith('.tif'):  # Check for TIFF files only
        #        img = tif.imread(os.path.join(path, file_name))  # Read the image
        #        self.data.append(img)  # Append the image to the list

        if len(self.data) == 0:  # If no TIFF images found, notify the user
            print("No TIFF files found.")
            return

        self.data = np.array(self.data)  # Convert list of images to a NumPy array

    def initUI(self):

        layout = QtWidgets.QGridLayout(self)  # Create a grid layout for arranging widgets

        self.segmentedContours = pg.ImageView()
        self.segmentedContours.ui.roiBtn.hide()
        self.segmentedContours.ui.roiPlot.hide()
        self.segmentedContours.ui.menuBtn.hide()
        self.segmentedContours.ui.histogram.hide()
        self.segmentedContours.setObjectName("segmentedContours")
        self.segmentedContours.getView().invertY(True)
        self.segmentedContours.getView().setMouseEnabled(x=False, y=False)


        

        self.imageView = pg.ImageView()  # Widget for displaying images
        layout.addWidget(self.imageView, 0, 0, 1, 3)  # Add image viewer spanning 3 columns

        # Create buttons for different actions
        self.avgButton = QtWidgets.QPushButton('Average Image')
        self.dynamicButton = QtWidgets.QPushButton('Dynamic Image')
        self.changePathButton = QtWidgets.QPushButton('Change Path')

        self.saveStackButton = QtWidgets.QPushButton('Save TIFF Stack')

        self.saveROIStackButton = QtWidgets.QPushButton('Save ROI Stack')


        self.drawROIButton = QtWidgets.QPushButton('Draw ROI')

        # Connect button clicks to their respective functions
        self.avgButton.clicked.connect(self.showAverageImage)
        self.dynamicButton.clicked.connect(self.showDynamicImage)
        self.changePathButton.clicked.connect(self.changePath)
        self.drawROIButton.clicked.connect(self.createROI)
        self.saveStackButton.clicked.connect(self.saveTIFFStack)
        self.saveROIStackButton.clicked.connect(self.save_roi_stack)



        




        

        # Add buttons to layout
        layout.addWidget(self.avgButton, 1, 0)
        layout.addWidget(self.dynamicButton, 1, 1)
        layout.addWidget(self.changePathButton, 1, 2)
        layout.addWidget(self.drawROIButton, 2, 0)
        layout.addWidget(self.saveStackButton, 2, 1)
        layout.addWidget(self.saveROIStackButton, 2, 2)

        self.setLayout(layout)  # Set the layout for the main widget
        self.setWindowTitle('Segmentation Data Curator')  # Set window title
        self.show()  # Display the window
    
    
    def save_current_levels(self, target ='avg'):
        item = self.imageView.getImageItem()

        levels = item.getLevels() if item else None

        if levels is not None:
            levels_arr = np.array(levels)
            if levels_arr.ndim in [1,2]:

                if target == 'avg':
                    self.avg_levels = tuple(levels)
                elif target == 'dynamic':
                    self.dynamic_levels = tuple(levels) 


    def restore_levels(self, source ='avg'):
        item = self.imageView.getImageItem()
        levels = self.avg_levels if source == 'avg' else self.dynamic_levels


        if item is not None and levels is not None:
            levels = np.array(levels)
            if levels.ndim in [1, 2]:
                item.setLevels(levels.tolist() if isinstance(levels, np.ndarray) else levels)
    def showAverageImage(self):

        self.storeViewSettings('dynamic')

        

        if self.avg_frame is None: #if average image is not already computed, compute it

            if self.data.ndim == 3:  # If data is a stack of images
                self.avg_frame = np.mean(self.data, axis=0)  # Compute average along the time/frame axis
            else:
                self.avg_frame = self.data  # If already 2D, just use the data directly

        
        


        # levels = self.imageView.getImageItem().getLevels()

        # if levels is not None and isinstance(levels, (list, tuple)) and len(levels) == 2:
        #     self.dynamic_levels = levels



        self.imageView.setImage(self.avg_frame, autoLevels = False)  # Display the image

        self.restoreViewSettings('average')



    def showDynamicImage(self):


        self.storeViewSettings('average')

        if self.dynamic_image is not None:

            self.imageView.setImage(self.dynamic_image, autoLevels = False)  # Display the image

            self.restoreViewSettings('dynamic')
            return

        if self.data.ndim != 3:
            print("Cannot perform dynamic analysis on 2D data.")
            return
        


        if self.dynamic_image is None:
            frames = self.data[:min(2000, self.data.shape[0])]  # Use up to 3000 frames to avoid overload

            from scipy.signal import butter, filtfilt  # Import necessary functions

            # Design a high-pass Butterworth filter

            print("Butter")
            b, a = butter(2, 1, btype='high', fs=500)

            # Apply the filter along the time/frame axis

            print("Applying")
            frames = filtfilt(b, a, frames, axis=0)

            if frames.shape[1] > 10 and frames.shape[2] > 10:  # Check if cropping is safe
                # Crop 5 pixels from each edge to reduce noise/artifacts
                max_frame = np.max(frames, axis=0)
                med_frame = np.median(frames, axis=0)
                var_frame = np.var(frames, axis=0)
            else:
                # Use full frame if dimensions are small
                max_frame = np.max(frames, axis=0)
                med_frame = np.median(frames, axis=0)
                var_frame = np.var(frames, axis=0)
        
        
        self.dynamic_image = (max_frame - med_frame) * var_frame

        
        # levels = self.imageView.getImageItem().getLevels()

        # if levels is not None and isinstance(levels, (list, tuple)) and len(levels) == 2:
        #     self.avg_levels = levels

        
        self.imageView.setImage(self.dynamic_image, autoLevels = False)  # Display the result

        self.restoreViewSettings('dynamic')
        # if self.dynamic_levels is not None:
        #     self.imageView.getImageItem().setLevels(*self.dynamic_levels)

    def storeViewSettings(self, key):
        hist = self.imageView.getHistogramWidget()

        if hist:
            self.levels_dict[key] = hist.getLevels()
            self.gradient_dict[key] = hist.item.saveState()



    def padShape(self, img1, img2):

        shape1, shape2 = img1.shape, img2.shape

        target_shape = (max(shape1[0], shape2[0]), max(shape1[1], shape2[1]))

        def pad(img, target_shape):
            pad_height = target_shape[0] - img.shape[0]
            pad_width = target_shape[1] - img.shape[1]

            return np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant')


        return pad(img1, target_shape), pad(img2, target_shape)

    def saveTIFFStack(self):


        def to_uint8(img):
            img = img - np.min(img)
            img = img / np.ptp(img)
            return (img * 255).astype(np.uint8)
        if self.avg_frame is None or self.dynamic_image is None:
            QtWidgets.QMessageBox.warning(self, "Missing data", "Please compute the average and dynamic images first.")
            return
        
        avg_img_padded, dynamic_img_padded = self.padShape(self.avg_frame, self.dynamic_image)


        avg_img_padded = to_uint8(avg_img_padded)
        dynamic_img_padded = to_uint8(dynamic_img_padded)
        stack = np.stack([avg_img_padded, dynamic_img_padded], axis = 0)


        save_path, _= QtWidgets.QFileDialog.getSaveFileName(self, "Save TIFF Stack", "combined_stack.tif", "TIFF files (*.tif)")

        if save_path:
            import tifffile as tif
            tif.imwrite("output_stack.tif", stack)
            QtWidgets.QMessageBox.information(self, "Success", f"Stack saved to {save_path}")




    def save_roi_stack(self, save_path='roi_stack.tiff'):
        if not self.rois:
            print("None to save")
            return
        
        image_shape = self.data.shape[1:] if self.data.ndim == 3 else self.data.shape
        roi_stack = []

        for roi in self.rois:
            mask = np.zeros(image_shape, dtype=np.uint8)
            poly = QtGui.QPolygonF([QtCore.QPointF(x,y) for x, y in self.rois_points[roi]])

            for y in range(image_shape[0]):
                for x in range(image_shape[1]):
                    if poly.contains(QtCore.QPointF(x, y), QtCore.Qt.OddEvenFill):
                        mask[y, x] = 255
                        
            roi_stack.append(mask)
        
        roi_stack = np.array(roi_stack, axis=0)
        tif.imwrite(save_path, roi_stack.astype(np.uint8))
        print(f"ROI stack saved to {save_path}")

    def restoreViewSettings(self, key):
        hist = self.imageView.getHistogramWidget()

        if hist:
            levels = self.levels_dict.get(key)
            gradient = self.gradient_dict.get(key)

            if levels is not None:
                hist.setLevels(*levels)
            if gradient is not None:
                hist.item.restoreState(gradient)

    #Have right click: delete
    #Have left click: add point
    #Have double click: connect all the points
    def createROI(self):
        print("ROI Button clicked!")
        self.scatterItem = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0),
            hoverable=True,
            hoverBrush=pg.mkBrush(0, 255, 255)
        )
        self.imageView.getView().addItem(self.scatterItem)
        self.roi_coords = []
        self.rois = []
        self.rois_points = []
        self.proxy = pg.SignalProxy(self.imageView.getView().scene().sigMouseClicked, rateLimit=60, slot=self.draw_roi)


    def remove_roi(self, roi):
        for x, y in self.rois_points[roi]:
            self.remove_point(x, y)
        self.rois.remove(roi)
        self.imageView.getView().removeItem(roi)
        del self.rois_points[roi]

        
        self.rois_points = {self.rois[i]: self.rois_points[self.rois[i]] for i in range(len(self.rois))}

    def remove_point(self, x, y):
        x_val, y_val = self.scatterItem.getData()

        mask = np.sqrt((x_val - x)**2 + (y_val - y)**2) > 1
        #self.scatterItem.clear()

        self.scatterItem.setData(x_val[mask], y_val[mask])

    def draw_roi(self, event):

        
        scene_coords = event[0].scenePos()
        self.colors = ['r', 'b', 'g', 'y', 'm', 'c']

        if event[0].double():
            # pyqtgraph roi
            
            roi = pg.PolyLineROI(self.roi_coords, closed=True, pen=self.colors[len(self.rois) % len(self.colors)])
            self.rois.append(roi)
            self.imageView.getView().addItem(roi)
            self.rois_points[roi] = list(self.roi_coords)
            self.roi_coords = []

        else:
            img_coords = self.imageView.getView().mapSceneToView(scene_coords)
            remove_flag = False

            if event[0].button() == QtCore.Qt.RightButton:
                print("Right click detected")
                for roi in self.rois:
                    for handle in roi.getLocalHandlePositions():
                        pos = handle[1]  # Get the position as a QPointF object
                        y = pos[1] # Get the x and y coordinates
                        x = pos[0]
                        if np.sqrt((img_coords.x() - x)**2 + (img_coords.y() - y)**2) <= 8:
                            self.remove_roi(roi)
                            remove_flag = True
                            break
                        else:
                            print("Not close enough to remove ROI")
                            continue
                for coords in self.roi_coords:
                    if np.sqrt((img_coords.x() - coords[0])**2 + (img_coords.y() - coords[1]) ** 2) <= 8:
                        self.roi_coords.remove(coords)
                        self.remove_point(coords[0], coords[1])
                        remove_flag = True
                        break
                if remove_flag:
                    print("Removed", img_coords.x(), img_coords.y())
                    return

            if event[0].button() != QtCore.Qt.LeftButton:
                return
            
            for coords in self.roi_coords:
                if np.sqrt((img_coords.x() - coords[0])**2 + (img_coords.y() - coords[1])**2) <= 25:
                    print("Too close to existing point")
                    return
            

            self.roi_coords.append([img_coords.x(), img_coords.y()])
            self.scatterItem.addPoints([img_coords.x()], [img_coords.y()], pen=self.colors[len(self.rois) % len(self.colors)])
            print("Added Point", img_coords.x(), img_coords.y())



#Next Task

# one stack containing the ROIs (each one in its own frame)
#one collapsed stack into one image

#look into multithreading for where each thread takes a subpart of the frame (e.g. 1/4 of the image) and then combine them into one image
#it would be the last 2 functions in the gui.py (look into)



#FIX ROIs 



    def changePath(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')  # Open folder picker

        if folder_path:  # If a valid folder is selected
            self.loadData(folder_path)  # Load data from the new folder
            print(f"Loaded data from: {folder_path}")  # Notify user

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # Create the application instance
    ex = DataCuratorApp()  # Create and show the main app
    sys.exit(app.exec_())  # Start the event loop and exit when the app is closed


    
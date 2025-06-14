import numpy as np
from cellpose import models, io
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile


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

class CellposeRunner:
    """
    Wrapper class for running Cellpose with detailed configuration and output handling
    """
    def __init__(self, 
                 model_type="cyto3",
                 gpu=False,
                 channels=[0,0],
                 diameter=None,
                 flow_threshold=0.4,
                 cellprob_threshold=0.0,
                 normalize=True,
                 do_3D=False):
        """
        Initialize Cellpose pipeline
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('cyto3', 'nuclei', etc)
        gpu : bool 
            Whether to use GPU acceleration
        channels : list
            First channel is target, second is optional nuclear channel
            [0,0] = grayscale
            [1,0] = red
            [2,0] = green 
            [3,0] = blue
        diameter : float or None
            Expected diameter of cells in pixels. If None, will be estimated.
        flow_threshold : float
            Flow error threshold (0 to 1). Higher joins more cells.
        cellprob_threshold : float
            Cell probability threshold (-6 to 6). Higher gives fewer masks.
        normalize : bool
            Whether to normalize image intensities
        do_3D : bool
            Whether to process as 3D stack
        """
        # Initialize main Cellpose model
        self.model = models.Cellpose(
            gpu=gpu,
            model_type=model_type
        )
        
        # Store parameters
        self.channels = channels
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.normalize = normalize
        self.do_3D = do_3D
        
        # For storing results
        self.masks = None
        self.flows = None
        self.styles = None
        self.estimated_diameter = None
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        # Load image
        if isinstance(image_path, str):
            image_path = Path(image_path)
            
        if image_path.suffix == '.tif':
            img = tifffile.imread(str(image_path))
        else:
            img = io.imread(str(image_path))
            
        # Add channel axis if needed
        if img.ndim == 2:
            img = img[..., np.newaxis]
            
        return img
    
    def process_image(self, image):
        """
        Run full Cellpose pipeline on an image
        
        Parameters:
        -----------
        image : array
            Input image array
            
        Returns:
        --------
        dict containing:
            masks : array
                Integer mask array (0=background, 1,2,3...=cell labels)
            flows : list
                Flow fields and probabilities
            diameters : float
                Estimated or used diameter
        """
        # Run Cellpose
        masks, flows, styles, diams = self.model.eval(
            image,
            batch_size=1,
            channels=self.channels,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            normalize=self.normalize,
            do_3D=self.do_3D
        )
        
        # Store results
        self.masks = masks
        self.flows = flows
        self.styles = styles
        self.estimated_diameter = diams
        
        return {
            'masks': masks,
            'flows': flows,
            'diameters': diams
        }
    
    def analyze_results(self):
        """
        Analyze segmentation results
        
        Returns:
        --------
        dict containing cell statistics
        """
        if self.masks is None:
            raise ValueError("No masks found. Run process_image first.")
            
        # Get unique cell IDs (excluding background=0)
        cell_ids = np.unique(self.masks)[1:]
        
        # Calculate statistics for each cell
        stats = []
        for cell_id in cell_ids:
            mask = self.masks == cell_id
            props = {
                'cell_id': cell_id,
                'area': mask.sum(),
                'centroid': np.array(mask.nonzero()).mean(axis=1),
            }
            stats.append(props)
            
        return {
            'cell_count': len(cell_ids),
            'cell_stats': stats,
        }
    
    def save_results(self, output_dir, base_name):
        """
        Save segmentation results
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save results
        base_name : str
            Base name for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save masks
        tifffile.imwrite(
            output_dir / f"{base_name}_masks.tif",
            self.masks.astype(np.uint16)
        )
        
        # Save flows if available
        if self.flows is not None:
            flows_dict = {
                'flows': self.flows,
                'styles': self.styles,
                'estimated_diameter': self.estimated_diameter
            }
            np.save(output_dir / f"{base_name}_flows.npy", flows_dict)
            
    def visualize(self, image, output_path=None):
        """
        Create visualization of segmentation results
        
        Parameters:
        -----------
        image : array
            Original input image
        output_path : str or Path, optional
            If provided, save visualization to this path
        """
        if self.masks is None:
            raise ValueError("No masks found. Run process_image first.")
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original image
        if image.ndim == 3:
            # Use first channel if multiple channels
            ax1.imshow(image[..., 0], cmap='gray')
        else:
            ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image')
        
        # Plot masks
        from cellpose.utils import masks_to_outlines
        outlines = masks_to_outlines(self.masks)
        
        ax2.imshow(image[..., 0] if image.ndim == 3 else image, cmap='gray')
        ax2.imshow(outlines, cmap='Reds', alpha=0.5)
        ax2.set_title(f'Segmentation ({len(np.unique(self.masks))-1} cells)')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    runner = CellposeRunner(
        model_type="cyto3",
        channels=[0,0],  # Grayscale images
        diameter=30.0,   # Expected cell diameter in pixels
        gpu=False       # Use CPU
    )
    
    # Process single image
    image_path = "t17_23.tif"
    output_dir = "results"
    
    # Load and process
    image = runner.load_image(image_path)
    results = runner.process_image(image)
    
    # Analyze and save results
    stats = runner.analyze_results()
    print(f"Found {stats['cell_count']} cells")
    
    # Save results
    runner.save_results(output_dir, "my_image")
    
    # Visualize
    runner.visualize(image, output_path="segmentation_viz.png")
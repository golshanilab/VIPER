import numpy as np

# Load the .npy file with allow_pickle=True
data = np.load('C:\\Users\\sacaa\\Documents\\SchoolWork\\LabWork\\Voltage-Imaging-Processing\\results\\my_image_flows.npy', 
               allow_pickle=True)

# The data is stored as a dictionary, so you need to access it like this
data_dict = data.item()  # Convert from numpy object array to dictionary

# Now you can access different components
flows = data_dict['flows']
styles = data_dict['styles']
estimated_diameter = data_dict['estimated_diameter']

# Print information about the data
print("Available keys:", data_dict.keys())
print("\nFlows shape:", [f.shape for f in flows])
print("\nStyles shape:", styles.shape)
print("\nEstimated diameter:", estimated_diameter)

# If you want to visualize the flows
import matplotlib.pyplot as plt

# Plot flow visualization (first element of flows list is the RGB flow visualization)
plt.figure(figsize=(8, 8))
plt.imshow(flows[0])
plt.title('Flow Visualization')
plt.colorbar()
plt.show()

# Plot cell probability map (third element of flows list)
plt.figure(figsize=(8, 8))
plt.imshow(flows[2], cmap='gray')
plt.title('Cell Probability Map')
plt.colorbar()
plt.show()
import numpy as np

# Load the file
# data = np.load('./checkpoints/brain/G-losses.npz')
data = np.load('./checkpoints/brain-da/G-losses.npz')

# Display all arrays
for key in data:
    print(f"\n{key}:")
    print(data[key])
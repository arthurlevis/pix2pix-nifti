"""
Arthur Levisalles
"""

import numpy as np

# Load the file
data = np.load('path_to_G-losses.npz')  # path to experiment in checkpoints

# Display losses
for key in data:
    print(f"\n{key}:")
    print(data[key])
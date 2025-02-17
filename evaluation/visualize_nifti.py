"""
Arthur Levisalles
"""

import nibabel as nib
import matplotlib.pyplot as plt

# Load nifti file
# img = nib.load("../path_to_MRI")  # MRI
# img = nib.load("../path_to_realCT")  # real CT
img = nib.load("../path_to_sCT")  # sCT
data = img.get_fdata().astype(float)

# Slice number
slice_num = int(input("Enter slice number: "))

# Display slice 
plt.imshow(data[:,:,slice_num], cmap='gray')
plt.axis('off')
plt.show()
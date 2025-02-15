import nibabel as nib
import matplotlib.pyplot as plt

# Load nifti file
# img = nib.load("../1BB152/real_A_1BB152.nii.gz")  # MRI
# img = nib.load("../1BB152/real_B_1BB152.nii.gz")  # real CT
img = nib.load("../1BB152/fake_B_1BB152.nii.gz")  # sCT
data = img.get_fdata().astype(float)

# Slice number
slice_num = int(input("Enter slice number: "))

# Display slice 
plt.imshow(data[:,:,slice_num], cmap='gray')
plt.axis('off')
plt.show()
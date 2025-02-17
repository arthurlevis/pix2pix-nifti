"""
Arthur Levisalles
"""

import nibabel as nib

def count_slices(nii_path):
   img = nib.load(nii_path)
   shape = img.shape
   return {
      'sagittal': shape[0],
      'coronal': shape[1],
      'axial': shape[2]
   }

# Compare real & synthetic CT volumes
# realCT = '../path_to_realCT'
sCT = '.././path_to_sCT'

slices = count_slices(sCT)
print(f"Number of sagittal slices: {slices['sagittal']}")
print(f"Number of coronal slices: {slices['coronal']}")
print(f"Number of axial slices: {slices['axial']}")
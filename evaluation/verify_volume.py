import nibabel as nib

def count_slices(nii_path):
   img = nib.load(nii_path)
   shape = img.shape
   return {
      'sagittal': shape[0],
      'coronal': shape[1],
      'axial': shape[2]
   }

nii = '../brain-sample-paired/test/B/real_B_1BA001.nii.gz'

slices = count_slices(nii)
print(f"Number of sagittal slices: {slices['sagittal']}")
print(f"Number of coronal slices: {slices['coronal']}")
print(f"Number of axial slices: {slices['axial']}")
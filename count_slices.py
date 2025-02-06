import nibabel as nib

def count_slices(nii_path):
   img = nib.load(nii_path)
   shape = img.shape
   return {
      'sagittal': shape[0],
      'coronal': shape[1],
      'axial': shape[2]
   }

# # input
# nii_input = './results/brain-sample-test/test_latest/real_A_1BA054.nii.gz'
# slices_input = count_slices(nii_input)
# print(f"Number of sagittal slices: {slices_input['sagittal']}")
# print(f"Number of coronal slices: {slices_input['coronal']}")
# print(f"Number of axial slices: {slices_input['axial']}")

# output 
nii_output = './results/brain-sample-test/test_latest/fake_B_1BA054.nii.gz'
slices_output = count_slices(nii_output)
print(f"Number of sagittal slices: {slices_output['sagittal']}")
print(f"Number of coronal slices: {slices_output['coronal']}")
print(f"Number of axial slices: {slices_output['axial']}")
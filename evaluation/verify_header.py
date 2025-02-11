import SimpleITK as sitk
import numpy as np

real_path = '../brain-sample-paired/test/B/real_B_1BA001.nii.gz'
fake_path = '../results/brain-sample/test_latest/fake_B_1BA001.nii.gz'

real = sitk.ReadImage(real_path)
fake = sitk.ReadImage(fake_path)

print("Spacing:", real.GetSpacing() == fake.GetSpacing())
print("Origin:", real.GetOrigin() == fake.GetOrigin())
print("Direction:", real.GetDirection() == fake.GetDirection())  

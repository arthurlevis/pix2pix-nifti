"""
Arthur Levisalles
"""

import SimpleITK as sitk
import numpy as np

real_path = '../path_to_realCT'
fake_path = '../path_to_sCT'

real = sitk.ReadImage(real_path)
fake = sitk.ReadImage(fake_path)

# Must be True
print("Spacing:", real.GetSpacing() == fake.GetSpacing())
print("Origin:", real.GetOrigin() == fake.GetOrigin())
print("Direction:", real.GetDirection() == fake.GetDirection())  

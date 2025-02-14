import numpy as np
import SimpleITK as sitk
import os

def preprocess_volume(data, modality):
    """Preprocess volume: clip intensities & scale to [-1, 1]."""

    # Intensity clipping
    if modality == 'MR':
        p1, p99 = np.percentile(data, 1), np.percentile(data, 99)
        data = np.clip(data, p1, p99)
    elif modality == 'CT':
        data = np.clip(data, -1000, 2200) 

    # Intensity scaling to [-1, 1]
    data = np.interp(data, (data.min(), data.max()), (-1, 1))

    return data


def load_and_preprocess(file_path, modality):
    """Load NIfTI file, resample, & preprocess intensities."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load NIfTI file
    nii = sitk.ReadImage(file_path)

    # Convert to numpy array
    data = sitk.GetArrayFromImage(nii).astype(np.float32)

    # SimpleITK uses (z,y,x), while Nibabel uses (x,y,z)
    data = np.transpose(data, (2, 1, 0)) 

    # Preprocess intensities
    data = preprocess_volume(data, modality)

    return data

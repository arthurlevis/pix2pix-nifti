import numpy as np
import SimpleITK as sitk
import os

def resample_image(image, target_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkLinear):  # 1.0, 1.0, 2.5 for brain / 1.0, 1.0, 2.5 for pelvis
    """Resample image to target spacing using SimpleITK."""

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    target_size = [
        int(np.round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]
    resampled_image = sitk.Resample(
        image, target_size, sitk.Transform(), interpolator,
        image.GetOrigin(), target_spacing, image.GetDirection(), 0, image.GetPixelID()
    )
    return resampled_image


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


def load_and_preprocess_nifti(file_path, modality):
    """Load NIfTI file, resample, & preprocess intensities."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load NIfTI file
    image = sitk.ReadImage(file_path)

    # Resample to target spacing
    resampled_image = resample_image(image)

    # Convert to numpy array
    data = sitk.GetArrayFromImage(resampled_image).astype(np.float32)

    # Preprocess intensities
    data = preprocess_volume(data, modality)

    return data
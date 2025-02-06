import numpy as np
import nibabel as nib
import os
from collections import defaultdict

def reconstruct_volume(predictions, metadata, window_size=5, stride=4):
    volume_shape = metadata['shape']
    reconstructed = np.zeros(volume_shape)
    counts = np.zeros(volume_shape)
    
    for slice_start, pred_window in predictions.items():
        if hasattr(pred_window, 'cpu'):
            pred_window = pred_window.cpu().numpy()
        
        if pred_window.ndim == 4:
            pred_window = pred_window.squeeze(0)
        
        if pred_window.shape[1:] != (volume_shape[0], volume_shape[1]):
            resized_preds = []
            for i in range(pred_window.shape[0]):
                slice_2d = pred_window[i]
                resized_slice = resize_2d(slice_2d, (volume_shape[0], volume_shape[1]))
                resized_preds.append(resized_slice)
            pred_window = np.stack(resized_preds)
        
        pred_window = np.transpose(pred_window, (1, 2, 0))
        
        end_idx = min(slice_start + window_size, volume_shape[2])
        valid_indices = slice(slice_start, end_idx)
        
        reconstructed[:, :, valid_indices] += pred_window
        counts[:, :, valid_indices] += 1
    
    mask = counts > 0
    reconstructed[mask] /= counts[mask]
    return reconstructed

def resize_2d(image, target_size):
    from scipy.ndimage import zoom
    zoom_factors = (target_size[0] / image.shape[0], 
                   target_size[1] / image.shape[1])
    return zoom(image, zoom_factors, order=1)

def denormalize_volume(volume):
    volume = (volume +1) * 1600
    volume = volume -1000  
    return volume

def save_reconstructed_nifti(reconstructed_volume, reference_path, output_path, denormalize=True):
    ref_img = nib.load(reference_path)
    if denormalize:
        reconstructed_volume = denormalize_volume(reconstructed_volume)
    reconstructed_volume = reconstructed_volume.astype(np.int16)
    new_img = nib.Nifti1Image(reconstructed_volume, ref_img.affine, ref_img.header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_img, output_path)


def process_and_save_predictions(model_output, patient_original_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    predictions_by_volume = defaultdict(dict)
    for pred_data in model_output:
        patient_id = os.path.basename(pred_data['path']).split('real_B_')[1].split('.nii.gz')[0]
        slice_start = pred_data['slice_start']
        predictions_by_volume[patient_id][slice_start] = pred_data['visuals']['fake_B']
    
    for patient_id, predictions in predictions_by_volume.items():
        # Get the original file path for this patient
        example_pred = next(p for p in model_output if 
                           os.path.basename(p['path']).split('real_B_')[1].split('.nii.gz')[0] == patient_id)
        patient_original_path = example_pred['path']
        orig_img = nib.load(patient_original_path)
        metadata = {
            'shape': orig_img.shape,
            'affine': orig_img.affine,
            'header': orig_img.header
        }
        reconstructed = reconstruct_volume(predictions, metadata)
        output_path = os.path.join(save_dir, f"pred_{patient_id}.nii.gz")
        save_reconstructed_nifti(reconstructed, patient_original_path, output_path, denormalize=True)
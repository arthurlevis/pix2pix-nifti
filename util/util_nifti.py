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
    """Denormalize CT volume directly from [-1, 1] to HU range."""

    # print(f"Before denorm - min: {volume.min()}, max: {volume.max()}")
    volume = (volume +1) * 1600
    volume = volume -1000  
    # print(f"After denorm - min: {volume.min()}, max: {volume.max()}")

    return volume

# def copy_original_nifti(src_path, dst_path):
#     """Copy original NIfTI file without modification."""
#     img = nib.load(src_path)
#     nib.save(img, dst_path)

def save_reconstructed_nifti(reconstructed_volume, reference_path, output_path, denormalize=True):
    ref_img = nib.load(reference_path)
    
    if denormalize:
        reconstructed_volume = denormalize_volume(reconstructed_volume)
    
    # Cast to int16 for CT values
    reconstructed_volume = reconstructed_volume.astype(np.int16)
    
    new_img = nib.Nifti1Image(reconstructed_volume, ref_img.affine, ref_img.header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_img, output_path)

def process_and_save_predictions(model_output, original_path, save_dir):
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # # Get paths for real images
    # dataroot = os.path.dirname(os.path.dirname(os.path.dirname(original_path)))
    # phase = os.path.basename(os.path.dirname(os.path.dirname(original_path)))
    
    orig_img = nib.load(original_path)
    metadata = {
        'shape': orig_img.shape,
        'affine': orig_img.affine,
        'header': orig_img.header
    }
    
    predictions_by_volume = defaultdict(dict)
    for pred_data in model_output:
        vol_id = os.path.basename(pred_data['path'])
        slice_start = pred_data['slice_start']
        predictions_by_volume[vol_id][slice_start] = pred_data['visuals']['fake_B']
        
    #     # Get paths for real images
    #     real_a_path = os.path.join(dataroot, phase, 'A', vol_id)
    #     real_b_path = os.path.join(dataroot, phase, 'B', vol_id)
        
    #     # Save real images by copying original files
    #     copy_original_nifti(real_a_path, os.path.join(save_dir, f"real_A_{vol_id}"))
    #     copy_original_nifti(real_b_path, os.path.join(save_dir, f"real_B_{vol_id}"))
    
    # Process and save fake images
    for vol_id, predictions in predictions_by_volume.items():
        reconstructed = reconstruct_volume(predictions, metadata)
        output_path = os.path.join(save_dir, f"fake_B_{vol_id}")
        save_reconstructed_nifti(reconstructed, original_path, output_path, denormalize=True)
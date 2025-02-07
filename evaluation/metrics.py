import os
import glob
import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter

def compute_masked_metrics(ct_path, sct_path, mask_path):
    """Compute MAE, PSNR, SSIM between CT and synthetic CT within mask"""

    # Load NIfTI files
    ct_img = nib.load(ct_path)
    sct_img = nib.load(sct_path)
    mask_img = nib.load(mask_path)
    
    ct_data = ct_img.get_fdata()
    sct_data = sct_img.get_fdata()
    mask_data = mask_img.get_fdata().astype(bool)
    
    assert ct_data.shape == sct_data.shape == mask_data.shape, "Shape mismatch"
    
    # Masked MAE
    mae = np.abs(ct_data[mask_data] - sct_data[mask_data]).mean()
    
    # Masked PSNR
    ct_clipped = np.clip(ct_data, -1024, 3000)
    sct_clipped = np.clip(sct_data, -1024, 3000)
    mse = np.mean((ct_clipped[mask_data] - sct_clipped[mask_data]) ** 2)
    Q = 3000 - (-1024)
    psnr = 10 * np.log10(Q**2 / mse) if mse != 0 else float('inf')
    
    # Masked SSIM
    ct_adj = np.clip(ct_data, -1024, 3000) + 1024  # [0, 4024]
    sct_adj = np.clip(sct_data, -1024, 3000) + 1024
    L = 4024
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    window_size = 7
    
    # Compute local statistics
    mu_x = uniform_filter(ct_adj, size=window_size)
    mu_y = uniform_filter(sct_adj, size=window_size)
    sigma_x_sq = uniform_filter(ct_adj**2, size=window_size) - mu_x**2
    sigma_y_sq = uniform_filter(sct_adj**2, size=window_size) - mu_y**2
    sigma_xy = uniform_filter(ct_adj * sct_adj, size=window_size) - mu_x * mu_y
    
    # SSIM map
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = np.clip(numerator / denominator, 0, 1)  # Ensure [0,1] range
    
    # Apply mask and average
    ssim = ssim_map[mask_data].mean()
    
    return {'MAE': mae, 'PSNR': psnr, 'SSIM': ssim}


def calculate_average_metrics():
    """Batch process all synthetic CTs and compute average metrics"""

    real_ct_dir = "../brain-sample-paired/test/B"
    mask_dir = "../brain-sample-paired/test_masks"
    fake_ct_dir = "../results/brain-sample-batch/test_latest"  # verify latest experiment name
    
    metrics = {'MAE': [], 'PSNR': [], 'SSIM': []}
    processed = []
    
    for fake_path in glob.glob(os.path.join(fake_ct_dir, "fake_B_*.nii.gz")):
        base = os.path.basename(fake_path)
        patient_id = base.split("fake_B_")[1].replace(".nii.gz", "")
        
        real_path = os.path.join(real_ct_dir, f"real_B_{patient_id}.nii.gz")
        mask_path = os.path.join(mask_dir, f"mask_{patient_id}.nii.gz")
        
        if not all(os.path.exists(p) for p in [real_path, mask_path]):
            print(f"Skipping {patient_id} - missing files")
            continue
            
        try:
            results = compute_masked_metrics(real_path, fake_path, mask_path)
            for k in metrics: metrics[k].append(results[k])
            processed.append(patient_id)
            print(f"{patient_id}: MAE={results['MAE']:.2f}, PSNR={results['PSNR']:.2f}, SSIM={results['SSIM']:.4f}")
        except Exception as e:
            print(f"Error in {patient_id}: {str(e)}")
    
    print("\n=== Final Results ===")
    print(f"Processed {len(processed)} cases")
    print(f"Average MAE:   {np.mean(metrics['MAE']):.2f} ± {np.std(metrics['MAE']):.2f}")
    print(f"Average PSNR:  {np.mean(metrics['PSNR']):.2f} ± {np.std(metrics['PSNR']):.2f} dB")
    print(f"Average SSIM:  {np.mean(metrics['SSIM']):.4f} ± {np.std(metrics['SSIM']):.4f}")

if __name__ == "__main__":
    calculate_average_metrics()
import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter

def compute_masked_metrics(ct_path, sct_path, mask_path):
    # Load NIfTI files
    ct_img = nib.load(ct_path)
    sct_img = nib.load(sct_path)
    mask_img = nib.load(mask_path)
    
    ct_data = ct_img.get_fdata()
    sct_data = sct_img.get_fdata()
    mask_data = mask_img.get_fdata().astype(bool)
    
    assert ct_data.shape == sct_data.shape == mask_data.shape, "Shapes must match"
    
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
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = uniform_filter(ct_adj**2, size=window_size) - mu_x**2
    sigma_y_sq = uniform_filter(sct_adj**2, size=window_size) - mu_y**2
    sigma_xy = uniform_filter(ct_adj * sct_adj, size=window_size) - mu_xy
    
    # SSIM map
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator
    
    # Apply mask and average
    ssim = ssim_map[mask_data].mean()
    
    return {'MAE': mae, 'PSNR': psnr, 'SSIM': ssim}

# Example usage
if __name__ == "__main__":

    realCT = 'brain-sample-paired/test/B/1BA054.nii.gz'
    fakeCT = 'results/brain-sample-test/test_latest/fake_B_1BA054.nii.gz'
    mask = 'mask.nii.gz'

    metrics = compute_masked_metrics(realCT, fakeCT, mask)
    
    print(f"Masked MAE: {metrics['MAE']:.4f}")
    print(f"Masked PSNR: {metrics['PSNR']:.2f} dB")
    print(f"Masked SSIM: {metrics['SSIM']:.4f}")
import SimpleITK
from evalutils.io import SimpleITKLoader
import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop
import os
import glob


class ImageMetrics():
    def __init__(self):
        # Use fixed wide dynamic range
        # self.dynamic_range = [-1024., 3000.]
        self.dynamic_range = [-1000., 2200.]  # same range than in data/preprocess_nifti.py
    
    def score_patient(self, ground_truth_path, predicted_path, mask_path):
        loader = SimpleITKLoader()
        gt = loader.load_image(ground_truth_path)
        pred = loader.load_image(predicted_path)
        mask = loader.load_image(mask_path)
        
        caster = SimpleITK.CastImageFilter()
        caster.SetOutputPixelType(SimpleITK.sitkFloat32)
        caster.SetNumberOfThreads(1)

        gt = caster.Execute(gt)
        pred = caster.Execute(pred)
        mask = caster.Execute(mask)
        
        # Get numpy array from SITK Image
        gt_array = SimpleITK.GetArrayFromImage(gt)
        pred_array = SimpleITK.GetArrayFromImage(pred)
        mask_array = SimpleITK.GetArrayFromImage(mask)
        
        # Calculate image metrics
        mae_value = self.mae(gt_array,
                             pred_array,
                             mask_array)
        
        psnr_value = self.psnr(gt_array,
                               pred_array,
                               mask_array,
                               use_population_range=True)
        
        ssim_value = self.ssim(gt_array,
                               pred_array, 
                               mask_array)  # instead of mask
        return {
            'mae': mae_value,
            'ssim': ssim_value,
            'psnr': psnr_value
        }
    
    def mae(self,
            gt: np.ndarray, 
            pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Mean Absolute Error (MAE)
    
        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
    
        Returns
        -------
        mae : float
            mean absolute error.
    
        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
        mae_value = np.sum(np.abs(gt*mask - pred*mask))/mask.sum() 
        return float(mae_value)
    
    
    def psnr(self,
             gt: np.ndarray, 
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None,
             use_population_range: Optional[bool] = False) -> float:
        """
        Compute Peak Signal to Noise Ratio metric (PSNR)
    
        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
        use_population_range : bool, optional
            When a predefined population wide dynamic range should be used.
            gt and pred will also be clipped to these values.
    
        Returns
        -------
        psnr : float
            Peak signal to noise ratio..
    
        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
        if use_population_range:
            dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]
            
            # Clip gt and pred to the dynamic range
            gt = np.where(gt < self.dynamic_range[0], self.dynamic_range[0], gt)
            gt = np.where(gt > self.dynamic_range[1], self.dynamic_range[1], gt)
            pred = np.where(pred < self.dynamic_range[0], self.dynamic_range[0], pred)
            pred = np.where(pred > self.dynamic_range[1], self.dynamic_range[1], pred)
        else:
            dynamic_range = gt.max()-gt.min()
            
        # apply mask
        gt = gt[mask==1]
        pred = pred[mask==1]
        psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range)
        return float(psnr_value)
    
    
    def ssim(self,
              gt: np.ndarray, 
              pred: np.ndarray,
              mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Structural Similarity Index Metric (SSIM)
    
        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
    
        Returns
        -------
        ssim : float
            structural similarity index metric.
    
        """
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        if mask is not None:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
            # Mask gt and pred
            gt = np.where(mask==0, min(self.dynamic_range), gt)
            pred = np.where(mask==0, min(self.dynamic_range), pred)

        # Make values non-negative
        if min(self.dynamic_range) < 0:
            gt = gt - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        # Set dynamic range for ssim calculation and calculate ssim_map per pixel
        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]
        ssim_value_full, ssim_map = structural_similarity(gt, pred, data_range=dynamic_range, full=True)

        if mask is not None:
            # Follow skimage implementation of calculating the mean value:  
            # crop(ssim_map, pad).mean(dtype=np.float64), with pad=3 by default.
            pad = 3
            ssim_value_masked  = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)
            return ssim_value_masked
        else:
            return ssim_value_full

if __name__=='__main__':

    # # Single nii.gz file
    # metrics = ImageMetrics()
    # ground_truth_path = "../brain-sample-paired/test/B/real_B_1BA005.nii.gz"
    # predicted_path = "../results/brain-sample/test_latest/fake_B_1BA005.nii.gz"
    # mask_path = "../brain-sample-paired/test_masks/mask_1BA005.nii.gz"
    # print(metrics.score_patient(ground_truth_path, predicted_path, mask_path))

    # Iterate over all nii.gz files
    metrics = ImageMetrics()
    gt_dir = "../brain-sample-paired/test/B"
    pred_dir = "../results/brain-sample/test_latest"
    mask_dir = "../brain-sample-paired/test_masks"
    
    # Get all patient IDs from ground truth files
    all_gt = glob.glob(os.path.join(gt_dir, "real_B_1[BP][ABC]*.nii.gz"))
    patient_ids = [os.path.basename(f).split("real_B_")[1].split(".nii.gz")[0] for f in all_gt]

    results = []
    for pid in patient_ids:
        gt_path = os.path.join(gt_dir, f"real_B_{pid}.nii.gz")
        pred_path = os.path.join(pred_dir, f"fake_B_{pid}.nii.gz")
        mask_path = os.path.join(mask_dir, f"mask_{pid}.nii.gz")
        
        if all(os.path.exists(f) for f in [gt_path, pred_path, mask_path]):
            results.append(metrics.score_patient(gt_path, pred_path, mask_path))

    # Calculate mean metrics
    mean_metrics = {
        'mae': np.mean([x['mae'] for x in results]),
        'ssim': np.mean([x['ssim'] for x in results]),
        'psnr': np.mean([x['psnr'] for x in results])
    }
    print("\nMean metrics across all patients:")
    print(mean_metrics)

    # Calculate median metrics
    median_metrics = {
        'mae': np.median([x['mae'] for x in results]),
        'ssim': np.median([x['ssim'] for x in results]),
        'psnr': np.median([x['psnr'] for x in results])
    }
    print("\nMedian metrics across all patients:")
    print(median_metrics)

    # Calculate max metrics
    max_mae = np.max([x['mae'] for x in results])
    max_mae_idx = np.argmax([x['mae'] for x in results])  # Get index of max MAE
    max_mae_pid = patient_ids[max_mae_idx]  # Get corresponding patient ID
    print(f"\nMaximum MAE ({max_mae:.4f}) occurred for patient: {max_mae_pid}")

    # Calculate max metrics
    min_mae = np.min([x['mae'] for x in results])
    min_mae_idx = np.argmin([x['mae'] for x in results])  # Get index of max MAE
    min_mae_pid = patient_ids[min_mae_idx]  # Get corresponding patient ID
    print(f"\nMinimum MAE ({min_mae:.4f}) occurred for patient: {min_mae_pid}")

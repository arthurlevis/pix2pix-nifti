import os
import glob
import nibabel as nib
import numpy as np
import torch
from data.base_dataset import BaseDataset
from data.preprocess_nifti import load_and_preprocess

class NiftiDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'B')
        self.A_paths = sorted(glob.glob(os.path.join(self.dir_A, '*.nii.gz*')))
        self.B_paths = sorted(glob.glob(os.path.join(self.dir_B, '*.nii.gz*')))

        assert len(self.A_paths) == len(self.B_paths)
    
        # Sliding window parameters
        self.window_size = 5
        self.stride = 3  # Overlap of 2
        
        # Create expanded dataset indices
        self.expanded_indices = self._create_sliding_indices()


    def _create_sliding_indices(self):
        expanded_indices = []
        for vol_idx, path in enumerate(self.A_paths):  
            vol = nib.load(path).get_fdata()
            n_slices = vol.shape[2]
            
            n_windows = (n_slices - self.window_size) // self.stride + 1
            start_indices = [i * self.stride for i in range(n_windows)]
            
            for start_idx in start_indices:
                expanded_indices.append((vol_idx, start_idx))
        
        return expanded_indices

    def __getitem__(self, index):
        # Get volume and starting slice indices
        vol_idx, start_slice = self.expanded_indices[index]
        
        A_path = self.A_paths[vol_idx]
        B_path = self.B_paths[vol_idx]

        # Load and preprocess volumes
        A_img = load_and_preprocess(A_path, 'MR')
        B_img = load_and_preprocess(B_path, 'CT')

        # Extract window of slices
        slice_indices = [min(i, A_img.shape[2]-1) for i in range(start_slice, start_slice + self.window_size)]
        A_slices = [A_img[:, :, idx] for idx in slice_indices]
        B_slices = [B_img[:, :, idx] for idx in slice_indices]

        # Stack slices
        A_stack = np.stack(A_slices, axis=0)
        B_stack = np.stack(B_slices, axis=0)

        # Convert to tensors
        A = torch.from_numpy(A_stack).float()
        B = torch.from_numpy(B_stack).float()

        # Resize to 256x256
        A = torch.nn.functional.interpolate(A.unsqueeze(0), size=(256, 256), 
                                          mode='bilinear', align_corners=False)
        B = torch.nn.functional.interpolate(B.unsqueeze(0), size=(256, 256), 
                                          mode='bilinear', align_corners=False)

        return {
            'A': A.squeeze(0),
            'B': B.squeeze(0),
            'A_paths': A_path,
            'B_paths': B_path,
            'slice_start': start_slice
        }

    def __len__(self):
        return len(self.expanded_indices)
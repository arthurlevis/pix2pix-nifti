import os
import glob
import nibabel as nib
import numpy as np
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import torchvision.transforms as transforms

class NiftiDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize the dataset.

        Args:
            opt (Option class): Stores all the experiment flags; needs to be a subclass of BaseOptions.
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'A')  # 'phase' can be 'train', 'test', etc.
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'B')  # 'phase' can be 'train', 'test', etc.
        
        self.A_paths = sorted(glob.glob(os.path.join(self.dir_A, '*.nii*')))
        self.B_paths = sorted(glob.glob(os.path.join(self.dir_B, '*.nii*')))

        # Ensure that the number of A and B images match
        assert len(self.A_paths) == len(self.B_paths), "The number of images in A and B directories must be the same."

        # Define transformations if needed (e.g., normalization, resizing)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy array to tensor
        ])

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        
        # Load NIfTI files
        A_img = nib.load(A_path).get_fdata()
        B_img = nib.load(B_path).get_fdata()

        # Extract a middle slice (assuming axial slices)
        slice_idx = A_img.shape[2] // 2
        A_slice = A_img[:, :, slice_idx]
        B_slice = B_img[:, :, slice_idx]

        # Convert slices to tensors and add batch and channel dimensions
        A = torch.from_numpy(A_slice).unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, H, W]
        B = torch.from_numpy(B_slice).unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, H, W]

        # Resize to 256x256 using bilinear interpolation
        A = torch.nn.functional.interpolate(A, size=(256, 256), mode='bilinear', align_corners=False)
        B = torch.nn.functional.interpolate(B, size=(256, 256), mode='bilinear', align_corners=False)

        # Normalize to [0, 1]
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())

        # # Debugging: Print tensor shapes
        # print(f"A shape: {A.shape}, B shape: {B.shape}")

        return {'A': A.squeeze(0), 'B': B.squeeze(0), 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
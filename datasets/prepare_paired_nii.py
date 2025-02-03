import os
import shutil
import random

def prep_pairs(patient_folders, output_dir, split_ratio):
    """
    Organizes MRI &  CT data into paired images for pix2pix.

    Args:
        patient_folders (list): List of paths to patient folders containing MRI & CT files.
        output_dir (str): Path to the directory where the pix2pix structure will be created.
        split_ratio (tuple): Train, validation, & test split ratios.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(os.path.join(output_dir, 'train', 'A'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'B'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'A'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'B'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'A'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'B'), exist_ok=True)

    # Shuffle patient folders for randomness
    random.shuffle(patient_folders)

    # Calculate split indices
    num_patients = len(patient_folders)
    train_end = int(num_patients * split_ratio[0])
    val_end = train_end + int(num_patients * split_ratio[1])

    # Split into train, val, test
    train_patients = patient_folders[:train_end]
    val_patients = patient_folders[train_end:val_end]
    test_patients = patient_folders[val_end:]

    # Function to copy files to the correct split folder
    def copy_files(patients, split):
        for patient in patients:
            patient_id = os.path.basename(patient)
            mri_file = os.path.join(patient, 'mr.nii')  # MRI file (A)
            ct_file = os.path.join(patient, 'ct.nii')   # CT file (B)

            if os.path.exists(mri_file) and os.path.exists(ct_file):
                # Copy MRI (A) & CT (B) files
                shutil.copy(mri_file, os.path.join(output_dir, split, 'A', f'{patient_id}.nii'))
                shutil.copy(ct_file, os.path.join(output_dir, split, 'B', f'{patient_id}.nii'))

    # Copy files for each split
    copy_files(train_patients, 'train')
    copy_files(val_patients, 'val')
    copy_files(test_patients, 'test')

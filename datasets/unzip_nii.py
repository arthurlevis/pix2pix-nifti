import os
import gzip
import shutil
import glob

def unzip_gz(directory):
    """
    Unzips all .nii.gz files in the subdirectories.
    Preserves the folder structure & removes the original .gz files.

    Args:
        directory (str): Path to the patient directory containing .nii.gz files.
    """
    # Loop through each folder & file in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                # Full path to the .nii.gz file
                gz_path = os.path.join(root, file)
                # Full path to the output .nii file (remove .gz)
                nii_path = os.path.join(root, file[:-3])

                # Unzip the file
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(nii_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove the original .gz file
                os.remove(gz_path)
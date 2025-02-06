import zipfile
import tempfile
from prepare_pairs import prep_pairs
import argparse
import glob
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-file", required=True, help="Path to zip file")
    parser.add_argument("--paired-dir", required=True, help="Path to paired data directory")
    parser.add_argument("--anatomy", choices=['brain', 'pelvis'], help="Anatomical region")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                       help="train/val/test split ratio")
    args = parser.parse_args()

    zip_base = os.path.basename(args.zip_file)[:-4]  # remove .zip suffix

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(args.zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Considers nested (Task1.zip) or flat (brain or plevis.zip) structure
        anatomy_path = os.path.join(temp_dir, zip_base)
        if args.anatomy:
            anatomy_path = os.path.join(anatomy_path, args.anatomy)
            
        patient_folders = glob.glob(os.path.join(anatomy_path, '1[BP][ABC][0-9]*'))
        
        prep_pairs(
            patient_folders,
            args.paired_dir,
            tuple(args.split_ratio)
        )

        # Create masks directory for test patients
        test_dir = os.path.join(args.paired_dir, 'test', 'A')
        test_patients = [f.split('real_A_')[1].split('.nii.gz')[0] for f in os.listdir(test_dir)]
        mask_dir = os.path.join(args.paired_dir, 'test_masks')
        os.makedirs(mask_dir, exist_ok=True)

        for patient in test_patients:
            mask_file = os.path.join(anatomy_path, patient, 'mask.nii.gz')
            if os.path.exists(mask_file):
                shutil.copy(mask_file, os.path.join(mask_dir, f'mask_{patient}.nii.gz'))

if __name__ == '__main__':
    main()

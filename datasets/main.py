import os
from unzip_patient_nii import unzip_gz
from prepare_paired_nii import prep_pairs
import argparse
import glob

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--sample_dir", required=True, help="Path to pelvis sample directory")
   parser.add_argument("--output_dir", required=True, help="Path to output paired data directory") 
   parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.7, 0.2, 0.1],
                      help="train/val/test split ratio (e.g. 0.7 0.2 0.1)")
   args = parser.parse_args()

   unzip_gz(args.sample_dir)

   prep_pairs(
       glob.glob(f'{args.sample_dir}/1[P][ABC][0-9]*'),
       args.output_dir,
       tuple(args.split_ratio)
   )

if __name__ == '__main__':
   main()
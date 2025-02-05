import zipfile
import tempfile
from prepare_pairs import prep_pairs
import argparse
import glob
import os

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--zip-file", required=True, help="Path to zip file")
   parser.add_argument("--paired-dir", required=True, help="Path to paired data directory") 
   parser.add_argument("--split-ratio", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                      help="train/val/test split ratio")
   args = parser.parse_args()

   zip_base = os.path.basename(args.zip_file)[:-4]  # remove .zip suffix

   with tempfile.TemporaryDirectory() as temp_dir:
       with zipfile.ZipFile(args.zip_file, 'r') as zip_ref:
           zip_ref.extractall(temp_dir)
       prep_pairs(
           glob.glob(f'{temp_dir}/{zip_base}/1[BP][ABC][0-9]*'),
           args.paired_dir,
           tuple(args.split_ratio)
       )

if __name__ == '__main__':
   main()
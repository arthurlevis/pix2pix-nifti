### Steps to prepare dataset for pix2pix (SynthRAD2023 example)

- Add Task1.zip file to root directory `pix2pix-nifti`

- Activate conda envirnoment & run from this directory:

Replace 'region' by brain or pelvis

```bash
python main.py --zip-file ../Task1.zip --paired-dir ../<region>-paired --anatomy '<region>'
```

- default `--split-ratio` = [0.7, 0.15, 0.15]
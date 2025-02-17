This repository was adapted from pytorch-CycleGAN-and-pix2pix: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master

## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/arthurlevis/pix2pix-nifti
cd pytorch-pix2pix-nifti
```

- Install conda environment:
```bash
`conda env create -f environment.yml`
```

- Follow steps in `/datasets` to prepare dataset for pix2pix-nifti.

- Train
```bash
# on CPU (check available options)
python train.py --dataroot ./path_to_paired_dataset --name train-name --model pix2pix

# on GPU
qsub train.job
```

- Test
```bash
# on CPU (check available options)
python test.py --dataroot ./path_to_paired_dataset --name test-name --model pix2pix

# on GPU
qsub test.job
```
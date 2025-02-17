## Overview of Code Structure

[train.py](../train.py) is a general-purpose training script. See the main [README](.../README.md) and [training/test  tips](tips.md) for more details.

[test.py](../test.py) is a general-purpose test script. Once you have trained your model with `train.py`, you can use this script to test the model. It will load a saved model from `--checkpoints_dir` and save the results to `--results_dir`. See the main [README](.../README.md) and [training/test tips](tips.md) for more details.

[data](../data) directory contains the modules related to the loading and preprocessing of NIfTI files. 

* [\_\_init\_\_.py](../data/__init__.py) implements the interface between this package and training and test scripts. `train.py` and `test.py` call `from data import create_dataset` and `dataset = create_dataset(opt)` to create a dataset given the option `opt`.
* [base_dataset.py](../data/base_dataset.py) implements an abstract base class ([ABC](https://docs.python.org/3/library/abc.html)) for datasets. It also includes common transformation functions (e.g., `get_transform`, `__scale_width`), which can be later used in subclasses.
* [image_folder.py](../data/image_folder.py) implements an image folder class. We modify the official PyTorch image folder [code](https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py) so that this class can load images from both the current directory and its subdirectories.
* [preprocess_nifti.py](./data/preprocess_nifti.py) implements standard preprocessing for medical images (intensity clipping and rescaling).
* [nifti_dataset.py](../data/nifti_dataset.py) implements a sliding window approach that processes five consecutive axial slices with a two-slice overlap.


[models](../models) directory contains modules related to objective functions, optimizations, and network architectures. Below we explain each file in details.

* [\_\_init\_\_.py](../models/__init__.py)  implements the interface between this package and training and test scripts.  `train.py` and `test.py` call `from models import create_model` and `model = create_model(opt)` to create a model given the option `opt`. You also need to call `model.setup(opt)` to properly initialize the model.
* [base_model.py](../models/base_model.py) implements an abstract base class ([ABC](https://docs.python.org/3/library/abc.html)) for models. It also includes commonly used helper functions (e.g., `setup`, `test`, `update_learning_rate`, `save_networks`, `load_networks`), which can be later used in subclasses.
* [pix2pix_model.py](../models/pix2pix_model.py) implements the pix2pix [model](https://phillipi.github.io/pix2pix/), for learning a mapping from input images to output images given paired data. It was adapted for NIfTI files. The model training requires `--dataset_mode nifti` dataset. 
* [networks.py](../models/networks.py) module implements network architectures (both generators and discriminators), as well as normalization layers, initialization methods, optimization scheduler (i.e., learning rate policy), and GAN objective function (`vanilla`, `lsgan`, `wgangp`).

[options](../options) directory includes our option modules: training options, test options, and basic options (used in both training and test). `TrainOptions` and `TestOptions` are both subclasses of `BaseOptions`. They will reuse the options defined in `BaseOptions`.
* [\_\_init\_\_.py](../options/__init__.py)  is required to make Python treat the directory `options` as containing packages,
* [base_options.py](../options/base_options.py) includes options that are used in both training and test. It also implements a few helper functions such as parsing, printing, and saving the options. It also gathers additional options defined in `modify_commandline_options` functions in both dataset class and model class.
* [train_options.py](../options/train_options.py) includes options that are only used during training time.
* [test_options.py](../options/test_options.py) includes options that are only used during test time.


[util](../util) directory includes a miscellaneous collection of useful helper functions.
  * [\_\_init\_\_.py](../util/__init__.py) is required to make Python treat the directory `util` as containing packages,
  * [util.py](../util/util.py) consists of simple helper functions such as `tensor2im` (convert a tensor array to a numpy image array), `diagnose_network` (calculate and print the mean of average absolute value of gradients), and `mkdirs` (create multiple directories).
  * [util_nifti.py](../util/util_nifti.py) reconstructs model output in the original input volume and saves predictions as compressed NIfTI files.
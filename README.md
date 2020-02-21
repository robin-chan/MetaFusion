This framework has not yet been tested to work out-of-the-box.

## What is MetaFusion:
TBD.

## Preparation:
We assume that the user is already using a neural network for semantic segmentation and a corresponding dataset. For each image from the segmentation dataset, MetaFusion requires a hdf5 file that contains the following data:

- a three-dimensional numpy array (height, width, classes) that contains the softmax probabilities computed for the current image
- the full filepath of the current input image
- a two-dimensional numpy array (height, width) that contains the ground truth class indices for the current image

The class object in "prepare_data.py" may help to generate these hdf5 files (possibly needs to be updated).

## Run Code:
```sh
./x.sh
```

## Packages and their versions we used:
We used Python 3.6.5. Please make sure to have the following packages installed:

- Cython==0.29.13
- h5py==2.10.0
- matplotlib==3.0.3
- numpy==1.17.2
- pandas==0.24.2
- Pillow==6.1.0
- scikit-learn==0.21.3

See also requirements.txt.

## Authors:
Robin Chan (University of Wuppertal) and others...

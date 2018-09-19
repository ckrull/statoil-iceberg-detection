### README

## Iceberg detection - Capstone Project 

### Cornelius Krull - Udacity machine learning nanodegree - July 2018

following files are included in this submission:

## Report
- `capstone_report_krull.pdf` The capstone report 
- `proposal.pdf` The capstone proposal
## Jupyter Notebooks
- `1_Data_Exploration.ipynb` The initial exploration of the data
- `2_Training_simple_CNN-k-fold-CV.ipynb` k-fold cross validation grid search to find the best CNN from scratch 
- `3_Keras_transfer_learning.ipynb` Transfer learning of ImageNet based models
- `4_Training_Scikitlearn_models.ipynb` exploration of Scikit learn models
- `5_Data_Augmentation.ipynb` - data augmentation to improve performance of CNN
- `6_Soft_voting_ensemble.ipynb` - Soft voting implementation and calculation of test submission

## Python source code
- `kfold_keras.py` implementation of kfold crossvalidation for keras models
- `helper.py`  various small helper functions used throughout the project
- `statoil_models.py` The convolutional neural networks used in this project
- `guidedfilter.py` A python implementation of the guided filter
## Bottleneck features

- `Xception_bottle.h5` Bottleneck features for Xception
- `VGG19_bottle.h5` Bottleneck features for VGG19
- `VGG16_bottle.h5`Bottleneck features for VGG16

## Dependencies

The following libraries were used during the completion of this project:

- `Keras` - Deep learning framework

- `Pandas`- Timeseries and Dataframe library

- `Numpy`- essential array library

- `Scipy`- Scientific computing package for python

- `Scikit-learn`- Machine learning library

- `Scikit-image`- Image processing in python

- `tqdm`- progressbars ( can be downloaded from here https://github.com/tqdm/tqdm)

## Datasets

- `train.json.7z`
- `test.json.7z`

The Dataset can also be downloaded from Kaggle:

https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data"# statoil-iceberg-detection" 

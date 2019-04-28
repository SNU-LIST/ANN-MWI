## Artificial Neural Network for Myelin Water Imaging: ANN-MWI
Source code to train and test the network (.py), and data preprocessing (.m) from the manuscript "Artificial neural network for myelin water imaging".



Overview
---------
![overview_fig](https://user-images.githubusercontent.com/49778751/56742469-c4311700-67af-11e9-84ec-b89c651bf189.png)

Two different artificial neural networks (ANN I and ANN II) for generating myelin water imaging were proposed in the manuscript.

In the *trained_networks* folder, final parameters from three different networks were uploaded.

(Three networks: *ANNI_mwf* for myelin water fraction, *ANNI_gmt2* for geometric mean T2, and *ANNII* for T2 distribution)

Please read the usage below for details.



Requirements
---------
* Python 2.7

* TensorFlow 1.9.0

* NVIDIA GPU (CUDA 8.0)

* MATLAB R2017b



Usage
---------
### Training

- **make_trainingset.m**: to make the training set with normalizing data.

- **train_ANNMWI.py**: to train the network with the normalized training set.



### Test

- **make_testset.m**: to make the test set with normalizing data.

- **test_ANNMWI.py**: to test the trained network with the normalized test set.

     For the test, you can use our results (.ckpt files) in each corresponding network folder in the *trained_networks* folder.
               


References
---------
Please cite the following publication if you use our work.

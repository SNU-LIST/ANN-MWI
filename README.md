## Artificial Neural Network for Myelin Water Imaging: ANN-MWI
Source code to train and test the network (.py), and data preprocessing (.m) from the manuscript "Artificial neural network for myelin water imaging", submitted to Magnetic Resonance in Medicine.



Overview
---------
![github-190919-resize](https://user-images.githubusercontent.com/49778751/65219459-913b4700-daf3-11e9-9398-9ab0de7faef8.png)

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


Data acquisition
---------
* 3T Trio MRI scanner (Siemens Healthcare, Erlangen, Germany)

* 3D multi-combined gradient and spin echo sequence


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
Please cite the following publication if you use our work (https://doi.org/10.1002/mrm.28038).

       @ARTICLE{
       author = {{Lee}, Jieun and {Lee}, Doohee and {Choi}, Joon Yul and
       {Shin}, Dongmyung and {Shin}, Hyeong-Geol and {Lee}, Jongho},
       title = "{Artificial neural network for myelin water imaging}",
       journal = {Magnetic Resonance in Medicine},
       year = "2020",
       month = "May",
       eid = {Volume 83, Issue 5},
       pages = {1875-1883},
       }

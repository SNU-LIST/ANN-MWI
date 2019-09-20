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
Please cite the following publication if you use our work (http://arxiv.org/abs/1904.12522).

       @ARTICLE{2019arXiv190412522L,
       author = {{Lee}, Jieun and {Lee}, Doohee and {Choi}, Joon Yul and
       {Shin}, Dongmyung and {Shin}, Hyeong-Geol and {Lee}, Jongho},
       title = "{Artificial neural network for myelin water imaging}",
       journal = {arXiv e-prints},
       keywords = {Electrical Engineering and Systems Science - Image and Video Processing},
       year = "2019",
       month = "Apr",
       eid = {arXiv:1904.12522},
       pages = {arXiv:1904.12522},
       archivePrefix = {arXiv},
       eprint = {1904.12522},
       primaryClass = {eess.IV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190412522L},
       adsnote = {Provided by the SAO/NASA Astrophysics Data System}
       }

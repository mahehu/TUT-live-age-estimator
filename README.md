# TUT live age estimator

**Python implementation of a live deep learning based age/gender/expression recognizer.**

![Image](doc/demo.jpg "demo")

All components use convolutional networks:

 * Detection uses an SSD model trained on Tensorflow object detection API, but running on OpenCV.
 * Age, gender and facial expression use a mobilenet trained and running on keras.
 
The detailed functionality of the system is described in our paper:

>Janne Tommola, Pedram Ghazi, Bishwo Adhikari, Heikki Huttunen, "[Real Time System for Facial Analysis](https://arxiv.org/abs/1809.05474)," Submitted to EUVIP2018.

If you use our work for research purposes, consider citing the above work.

## Usage instructions:


Dependencies: [OpenCV 3.4.1+](http://www.opencv.org/), [Tensorflow 1.8+](http://tensorflow.org), [Keras 2.2.2+](http://keras.io/) and [dlib 19.4+](http://dlib.net/).

  * Requires a webcam.
  * Tested on Ubuntu Linux 16.04, 18.04 and Windows 10 with and without a GPU.
  * Install opencv 3.4.1 or newer. Recommended to install with `pip3 install opencv-python` (includes GTK support, which is required).
  * Install Tensorflow (1.8 or newer). On a CPU, the MKL version seems to be radically faster than others (Anaconda install by smth like `conda install tensorflow=1.10.0=mkl_py36hb361250_0`. Seek for proper versions with `conda search tensorflow`.). On GPU, use `pip3 install tensorflow-gpu`. Note that CPU is probably faster.
  * Install Keras 2.2.2 (or newer). Earlier versions have a slightly different way of loading the models. For example: `pip3 install keras`.
  * Install dlib (version 19.4 or newer) with python 3 dependencies; _e.g.,_ `pip3 install dlib`.
  * Download the required deep learning models from [here](http://www.cs.tut.fi/~hehu/models.zip) [[mirror link](https://tutfi-my.sharepoint.com/:u:/g/personal/janne_tommola_tut_fi/EcrQbRgnsydApRFsmsUbPfABcEK0arXtCe796Bt1x7_U7g?e=fQJN7Z)]. Extract directly to the main folder so that 2 new folders are created there.
  * Run with `python3 EstimateAge.py`.


Example video [here](https://youtu.be/Kfe5hKNwrCU).

Contributors: [Heikki Huttunen](http://www.cs.tut.fi/~hehu/), Janne Tommola

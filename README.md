# TUT live age estimator

**Python implementation of a live deep learning based age/gender/smile/celebrity twin recognizer.**

![Image](doc/demo.jpg "demo")

All components use convolutional networks:

 * Detection uses an SSD model trained on Tensorflow object detection API, but running on OpenCV.
 * Age, gender, and smile recognition use a multitask mobilenet trained and running on keras.
 * Celebrity twin uses a squeeze-excite seresnet18 to extract features, trained and running on keras.
 
The detailed functionality of the system (without multitask and celebrity similarity) is described in our paper:

>Janne Tommola, Pedram Ghazi, Bishwo Adhikari, Heikki Huttunen, "[Real Time System for Facial Analysis](https://arxiv.org/abs/1809.05474)," Submitted to EUVIP2018.

If you use our work for research purposes, consider citing the above work.

## Usage instructions:


Dependencies: [OpenCV 4.0.1+](http://www.opencv.org/), [Tensorflow 1.8+](http://tensorflow.org), [Keras 2.2.3+](http://keras.io/), and [faiss](https://github.com/facebookresearch/faiss/).

  * Requires a webcam.
  * Tested on Ubuntu Linux 16.04, 18.04 and Windows 10 with and without a GPU.
  * Install OpenCV 4.0.1 or newer. Recommended to install with `pip3 install opencv-python` (includes GTK support, which is required). Freetype support for nicer fonts requires manual compilation of OpenCV.
  * Install Tensorflow (1.8 or newer). On a CPU, the MKL version seems to be radically faster than others (Anaconda install by smth like `conda install tensorflow=1.10.0=mkl_py36hb361250_0`. Seek for proper versions with `conda search tensorflow`.). On GPU, use `pip3 install tensorflow-gpu`.
  * Install Keras 2.2.3 (or newer). Earlier versions have a slightly different way of loading the models. For example: `pip3 install keras`.
  * Install dlib (version 19.4 or newer) with python 3 dependencies; _e.g.,_ `pip3 install dlib`.
  * Install faiss with Anaconda `conda install faiss-cpu -c pytorch`.
  * Run with `python3 EstimateAge.py`.

[Required deep learning models and celebrity dataset](http://doi.org/10.5281/zenodo.3466980). Extract directly to the main folder so that 2 new folders are created there.

[Example video](https://youtu.be/Kfe5hKNwrCU).

Contributors: [Heikki Huttunen](http://www.cs.tut.fi/~hehu/), Janne Tommola

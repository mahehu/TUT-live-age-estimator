# TUT-live-age-estimator

## Python implementation of a live deep learning based age/gender/expression recognizer.

*2018-09-18*: Updated detection and recognition to use modern Mobilenets through OpenCV and Keras, removed Caffe dependencies. Aligner updated. Other minor changes.

![Image](doc/demo.jpg "demo")

The functionality of the system is described in our paper:

Janne Tommola, Pedram Ghazi, Bishwo Adhikari, Heikki Huttunen, "Real Time System for Facial Analysis," Submitted to EUVIP2018. [ [arXiv link](https://arxiv.org/abs/1809.05474]) ]

If you use our work for research purposes, consider citing the above work.

## Usage instructions:

  * Requires a webcam.
  * Install opencv 3.4.1 or newer. Recommended to install with `pip install opencv-python` (includes GTK support, which is required).
  * Keras 2.2.2 (or newer). Earlier versions have a slightly different way of loading the models.
  * Download the required deep learning models from [here](http://www.cs.tut.fi/~hehu/models.zip) or [here](https://tutfi-my.sharepoint.com/:u:/g/personal/janne_tommola_tut_fi/EcrQbRgnsydApRFsmsUbPfABcEK0arXtCe796Bt1x7_U7g?e=fQJN7Z). Extract directly to the main folder so that 2 new folders are created there.
  * Run with `python EstimateAge.py`.


Example video of <b>old version</b> <a href="https://youtu.be/Kfe5hKNwrCU">here</a>.



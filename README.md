# TUT-live-age-estimator
Python implementation of a live deep learning based age/gender/expression recognizer.

<b>2018-09-18</b>: Updated detection and recognition to use modern Mobilenets through OpenCV and Keras, removed Caffe dependencies. Aligner updated. Other minor changes.

<p align="center">
  <img src="doc/demo.jpg" width=544 height=306>
  <br>
   The system in a test environment
</p>
<br><br>

The functionality of the system is described in our paper:<br>
Janne Tommola, Pedram Ghazi, Bishwo Adhikari, Heikki Huttunen, "Real Time System for Facial Analysis," Submitted to EUVIP2018. [<a href="Real Time System for Facial Analysis">arXiv link</a>].<br>
If you use our work for research purposes, consider citing the above work.

<b>Usage instructions: </b>

<ul>
  <li>Requires a webcam.</li>
  <li> Install opencv 3.4.1 or newer. Recommended to install with `pip install opencv-python` (includes GTK support, which is required).</li>
  <li>Keras 2.2.2 (or newer). Earlier versions have a slightly different way of loading the models.</li>
<li>Download the required deep learning models from <a href="https://tutfi-my.sharepoint.com/:u:/g/personal/janne_tommola_tut_fi/EcrQbRgnsydApRFsmsUbPfABcEK0arXtCe796Bt1x7_U7g?e=fQJN7Z">here</a> or <a href="http://www.cs.tut.fi/~hehu/models.zip">here</a>. Extract directly to the main folder so that 2 new folders are created there.</li>
  <li>Run with `python EstimateAge.py`</li>
</ul>

Example video of <b>old version</b> <a href="https://youtu.be/Kfe5hKNwrCU">here</a>.



# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:47:04 2018

@author: agedemo
"""

from keras.models import load_model
import h5py
import glob
import cv2
import sys
import numpy as np
import time
import os
import tensorflow as tf
from keras import backend as K 

visualize = True
if visualize:
    import matplotlib.pyplot as plt

def find_images_from_tree(path):
    """ Collect images from a tree with one folder per identity """

    print("Searching for images in {}".format(path))
    image_files = []
    
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.lower().endswith(("jpg", "jpeg", "png", "bmp")):
                image_files.append(root + os.sep + name)
                
    return image_files

def find_images(path):
    """ Collect one image per identity """

    found_ids = []
    files = []

    top_folder = os.sep.join(path.split(os.sep)[:-1])
    identity_file = top_folder + os.sep + "identities.txt"

    with open(identity_file) as fp:
        for i, line in enumerate(fp):

            name, identity = line.split()
            identity = int(identity)
            
            if identity not in found_ids:
                found_ids.append(identity)
                fullfile = os.path.abspath(path + os.sep + os.path.basename(name))

                if not os.path.isfile(fullfile):
                    print("File {} not found, ignoring.".format(fullfile))
                else:
                    files.append(fullfile)

    return files

def triplet_semihard_loss(y_true, y_pred):

    loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels = K.argmax(y_true, axis = -1), embeddings = y_pred, margin = 1.0)
    return loss

def cluster_loss(y_true, y_pred):

    loss = tf.contrib.losses.metric_learning.cluster_loss(labels = K.argmax(y_true, axis = -1), embeddings = y_pred, margin_multiplier = 1.0)
    return loss

def triplet_loss(y_true, y_pred):

    loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels = K.argmax(y_true, axis = -1), embeddings = y_pred, margin = 1.0)
    return loss

def lifted_struct_loss(y_true, y_pred):

    loss = tf.contrib.losses.metric_learning.lifted_struct_loss(labels = K.argmax(y_true, axis = -1), embeddings = y_pred, margin = 1.0)
    return loss

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        model_folder = sys.argv[1]
    else:
        model_folder = "recognizers/celebrities/network/INCEPTIONRESNET_2018-10-18-18-14-13/"

    model = load_model(model_folder + os.sep + "feature_model.h5", custom_objects = {'triplet_loss': triplet_loss, 'lifted_struct_loss': lifted_struct_loss, 'cluster_loss': cluster_loss})

    if len(sys.argv) > 2:
        images_folder = sys.argv[2]
    else:
        images_folder = "recognizers/celebrities/data/FinnishCelebs"

    #files = find_images(path = images_folder)
    files = find_images_from_tree(path = images_folder)

    # Gather the file structure of the dataset, used when visualizing with different images than the ones features are calculated from
    commonpath = os.path.commonpath((files[0], images_folder))
    path_ends = [os.path.relpath(file, start=commonpath) for file in files]
    
    in_shape = model.input_shape[1:3]
    out_dim  = model.output_shape[-1]
        
    features = np.empty((len(files), out_dim))
                
    print("Found {} files...".format(len(files)))

    if visualize:
        fig, ax = plt.subplots(2, 1)

    start_time = time.time()
    buf_size = 1
    fb_shape = (buf_size, ) + model.input_shape[1:]
    frame_buffer = np.empty(fb_shape, dtype = np.float32)
    fb_idx = 0
    
    cnt = 0
    prev_sample = None

    for i, name in enumerate(files):
        
        print(name)
        img = cv2.imread(name)
        
        # Take center crop and scale to in-shape
        h, w, d = img.shape
            
        if w > h:
            c = w // 2
            x1 = c - h // 2
            x2 = x1 + h
            img = img[:, x1:x2, :]
        elif w < h:
            c = h // 2
            y1 = c - w // 2
            y2 = y1 + w
            img = img[y1:y2, :, :]
        img = cv2.resize(img, in_shape)

        # Convert to RGB and scale
        img = img[..., ::-1].astype(np.float32) / 255.0
        frame_buffer[fb_idx, ...] = img
        fb_idx += 1
        
        if fb_idx == buf_size:
            
            feat = model.predict(frame_buffer)
            elapsed_time = time.time() - start_time
            sec_per_frame = elapsed_time / (i+1)
            remaining_frames = len(files) - (i+1)
            remaining_time = remaining_frames * sec_per_frame
            remaining_time_mins = remaining_time / 60
            
            msg = "Computing features: {:.1f} % done [{:.1f} MB]. {:.1f} mins remaining".format(100*(i+1) / len(files),
                                                      sys.getsizeof(features) / 1024**2,
                                                      remaining_time_mins)
    
            print(msg, end = " ")
            print("File {}".format(name))
            
            if visualize:        
                
                ax[0].cla()

                f = feat[0,...]
                ax[0].plot(f)
                ax[0].set_title(msg)

                if prev_sample is not None:
                    ax[1].cla()
                    ax[1].plot(f - prev_sample)
                    ax[1].set_title("Difference to previous sample")

                plt.show(block = False)
                plt.pause(0.1)
                prev_sample = f

            fb_idx = 0
        else:
            continue
            
        features[cnt : cnt + feat.shape[0], :] = feat
        cnt += feat.shape[0]
                            
    with h5py.File(model_folder + os.sep + "features_" + os.path.basename(os.path.normpath(images_folder)) + ".h5", "w") as h5file:
        h5file["features"]  = np.array(features)
        b_files = [bytes(f, 'utf-8') for f in files]
        h5file["filenames"] = b_files
        b_pathends = [bytes(f, 'utf-8') for f in path_ends]
        h5file["path_ends"] = b_pathends
        print(b_pathends)


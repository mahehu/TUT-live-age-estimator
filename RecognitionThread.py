#!/usr/bin/env python

import threading
import time
import numpy as np
import os
import glob
import caffe
import cv2
import dlib 

class RecognitionThread(threading.Thread):

    def __init__(self, parent, params):

        print "Initializing recognition thread..."

        threading.Thread.__init__(self)
        self.parent = parent

        caffe.set_mode_gpu()

        # Initialize networks for Age, Gender and Expression
        # Always take the most recent prototxt and caffemodel from 
        # each folder

        # 0. Alignment

        self.useAligner      = bool(params.get("recognition", "use_aligner"))
        
        alignerPath          = params.get("recognition", "aligner")
        alignerTargetsPath   = params.get("recognition", "aligner_targets")
        self.aligner         = dlib.shape_predictor(alignerPath)
        self.aligner_targets = np.loadtxt(alignerTargetsPath)
        
        # 1. AGE

        dcnnPath = params.get("recognition", "age_folder")
        paramFiles = glob.glob(dcnnPath + os.sep + "*.caffemodel")
        paramFiles = sorted(paramFiles, key=lambda x: os.path.getctime(x))

        modelFiles = glob.glob(dcnnPath + os.sep + "*.prototxt")
        modelFiles = sorted(modelFiles, key=lambda x: os.path.getctime(x))
        
        MODEL_FILE = modelFiles[-1]
        PRETRAINED = paramFiles[-1]
        
        mean = cv2.imread(dcnnPath + os.sep + "imagenet_mean.jpg")
        mean = cv2.resize(mean, (224,224))
        mean = np.transpose(mean, (2,0,1))
        
        self.ageNet = caffe.Classifier(
            MODEL_FILE, PRETRAINED, image_dims=(256, 256), mean=mean)

        # 2. GENDER

        dcnnPath = params.get("recognition", "gender_folder")
        paramFiles = glob.glob(dcnnPath + os.sep + "*.caffemodel")
        paramFiles = sorted(paramFiles, key=lambda x: os.path.getctime(x))

        modelFiles = glob.glob(dcnnPath + os.sep + "*.prototxt")
        modelFiles = sorted(modelFiles, key=lambda x: os.path.getctime(x))
        
        MODEL_FILE = modelFiles[-1]
        PRETRAINED = paramFiles[-1]

        self.genderNet = caffe.Classifier(
            MODEL_FILE, PRETRAINED, image_dims=(256, 256), mean=mean)

        # 3. EXPRESSION

        dcnnPath = params.get("recognition", "expression_folder")
        paramFiles = glob.glob(dcnnPath + os.sep + "*.caffemodel")
        paramFiles = sorted(paramFiles, key=lambda x: os.path.getctime(x))

        modelFiles = glob.glob(dcnnPath + os.sep + "*.prototxt")
        modelFiles = sorted(modelFiles, key=lambda x: os.path.getctime(x))
        
        MODEL_FILE = modelFiles[-1]
        PRETRAINED = paramFiles[-1]

        self.expressionNet = caffe.Classifier(
            MODEL_FILE, PRETRAINED, image_dims=(256, 256))

        # Read class names
        self.expressions = []
        
        while True:
            try:
                key = "class%d" % len(self.expressions)
                name = params.get("expression", key)
                self.expressions.append(name)                
            except:
                break
        
        self.minDetections = int(params.get("recognition", "mindetections"))
        
        print("Recognition thread started...")
    
    def estimateRigidTransform(self, src, dst):

        H = np.vstack([src[:, 0], src[:, 1], np.ones_like(src[:, 0])]).T
        M = np.linalg.lstsq(H, dst)[0].T

        return M
        
    def run(self):
        
        caffe.set_mode_gpu()

        while self.parent.isTerminated() == False:

            faces = self.parent.getFaces()            
            
            while faces == None:
                time.sleep(0.1)
                faces = self.parent.getFaces()            

            validFaces = [f for f in faces if len(f['bboxes']) > self.minDetections]
            
            for face in validFaces:

                # get the timestamp of the most recent frame:

                timestamp = face['timestamps'][-1]
                
                img = self.parent.getUnit(self, timestamp).getFrame()
                mean_box = np.mean(face['bboxes'], axis = 0)
                x,y,w,h = [int(c) for c in mean_box]
                
                if self.useAligner:
                    
                    # Align the face to match the targets
                    
                    dlib_box = dlib.rectangle(left = x, top = y, right = x+w, bottom = y+h)
                    dlib_img = img[..., ::-1].astype(np.uint8)
                    
                    s = self.aligner(dlib_img, dlib_box)
    
                    landmarks = [[s.part(k).x, s.part(k).y] for k in range(s.num_parts)]
                    landmarks = np.array(landmarks)
                    M = self.estimateRigidTransform(landmarks, self.aligner_targets)
                    
                    landmarks = landmarks.astype(float)
                    landmarks -= np.min(landmarks, keepdims = True)
                    landmarks /= np.max(landmarks, keepdims = True)
                    landmarks *= (256 / 2)
                    landmarks += 0.5 * 256 / 2
                    landmarks[:, 1] += 40
                    
                    crop = cv2.warpAffine(img, M, (256,256))

                else:

                    # Add 40% margin around the face as in training time:
                    x -= 0.4*w
                    y -= 0.4*h
                    w += 0.8*w
                    h += 0.8*h
                    
                    x,y,w,h = [int(c) for c in [x,y,w,h]]
                    
                    # make sure inside image:
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if x + w > img.shape[1]:
                        x = img.shape[1] - w
                    if y + h > img.shape[0]:
                        y = img.shape[0] - h
                        
                    crop = img[y : y+h, x : x+w, ...]
              
                crop = crop.astype(np.float32)
                
                # Recognize age
                
                prob = self.ageNet.predict([crop], oversample=False).ravel()
                age = np.dot(prob, range(101))
                
                if "age" in face.keys():
                    face["age"] = 0.95 * face["age"] + 0.05 * age
                else:
                    face["age"] = age
                    
                # Recognize gender
                
                gender = self.genderNet.predict([crop], oversample=False).ravel()[1]
                if "gender" in face.keys():
                    face["gender"] = 0.99 * face["gender"] + 0.01 * gender
                else:
                    face["gender"] = gender

                # Recognize expression
                # For now it works on grayscale. Todo: train a proper net here
                
                gray = np.mean(crop, axis = -1, keepdims = True)
                
                out = self.expressionNet.predict([gray], oversample=False).ravel()
                expression = self.expressions[np.argmax(out)]
                face["expression"] = expression
            
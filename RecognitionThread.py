#!/usr/bin/env python

import threading
import time
import numpy as np
import os
from collections import namedtuple
import cv2
import dlib
import keras
from keras.utils.generic_utils import CustomObjectScope
from compute_features import lifted_struct_loss, triplet_loss
import h5py
import faiss


class RecognitionThread(threading.Thread):

    CELEB_RECOG_BUFFER = 15  # How many recognitions to store for picking the most common

    def __init__(self, parent, params):
        print("Initializing recognition thread...")
        threading.Thread.__init__(self)
        self.parent = parent

        ##### Initialize aligners for face alignment.
        aligner_path = params.get("recognition", "aligner")
        aligner_targets_path = params.get("recognition", "aligner_targets")
        self.aligner = dlib.shape_predictor(aligner_path)
        self.aligner_targets = np.loadtxt(aligner_targets_path)

        ##### Initialize networks for Age, Gender and Expression
        ##### 1. AGE
        print("Initializing age network...")
        agepath = params.get("recognition", "age_folder")
        with CustomObjectScope({'relu6': keras.layers.ReLU(6.),
                                'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
            self.ageNet = keras.models.load_model(os.path.join(agepath, 'model.h5'))
        self.ageNet._make_predict_function()

        ##### 2. GENDER
        print("Initializing gender network...")
        genderpath = params.get("recognition", "gender_folder")
        with CustomObjectScope({'relu6': keras.layers.ReLU(6), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
            self.genderNet = keras.models.load_model(os.path.join(genderpath, 'model.h5'))
        self.genderNet._make_predict_function()

        ##### 3. EXPRESSION
        print("Initializing expression network...")
        expressionpath = params.get("recognition", "expression_folder") 
        with CustomObjectScope({'relu6': keras.layers.ReLU(6.),
                                'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
            self.expressionNet = keras.models.load_model(os.path.join(expressionpath, 'model.h5'))
        self.expressionNet._make_predict_function()
        
        ##### Read class names
        self.expressions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
        self.minDetections = int(params.get("recognition", "mindetections"))

        ##### 4. CELEBRITY
        self.siamesepaths = params['celebmodels']
        self.siamesepath = self.siamesepaths["0"]
        self.celeb_dataset = params.get("recognition", "celeb_dataset")
        self.initialize_celeb()

        # Starting the thread
        self.switching_model = False
        self.recognition_running = False
        print("Recognition thread started...")

    def initialize_celeb(self):
        print("Initializing celebrity network...")

        with CustomObjectScope({'relu6': keras.layers.ReLU(6.),
                                'DepthwiseConv2D': keras.layers.DepthwiseConv2D,
                                'lifted_struct_loss': lifted_struct_loss,
                                'triplet_loss': triplet_loss}):
            self.siameseNet = keras.models.load_model(os.path.join(self.siamesepath, "feature_model.h5"))

        self.siameseNet._make_predict_function()

        ##### Read celebrity features
        celebrity_features = self.siamesepath + os.sep + "features_" + self.celeb_dataset + ".h5"
        print("Reading celebrity data from {}...".format(celebrity_features))

        with h5py.File(celebrity_features, "r") as h5:
            celeb_features = np.array(h5["features"]).astype(np.float32)
            self.celeb_files = list(h5["filenames"])
            self.celeb_files = [s.decode("utf-8") for s in self.celeb_files]

        print("Building index...")
        self.celeb_index = faiss.IndexFlatL2(celeb_features.shape[1])
        self.celeb_index.add(celeb_features)

    def estimateRigidTransform(self, landmarks, aligner_targets):
        # H = np.vstack([src[:, 0], src[:, 1], np.ones_like(src[:, 0])]).T
        # M = np.linalg.lstsq(H, dst)[0].T

        first_idx = 27
        B = aligner_targets[first_idx:, :]
        landmarks = landmarks[first_idx:]
        A = np.hstack((np.array(landmarks), np.ones((len(landmarks), 1))))

        a = np.row_stack((np.array([-A[0][1], -A[0][0], 0, -1]), np.array([
            A[0][0], -A[0][1], 1, 0])))
        b = np.row_stack((-B[0][1], B[0][0]))

        for j in range(A.shape[0] - 1):
            j += 1
            a = np.row_stack((a, np.array([-A[j][1], -A[j][0], 0, -1])))
            a = np.row_stack((a, np.array([A[j][0], -A[j][1], 1, 0])))
            b = np.row_stack((b, np.array([[-B[j][1]], [B[j][0]]])))
        X, res, rank, s = np.linalg.lstsq(a, b, rcond=-1)
        cos = (X[0][0]).real.astype(np.float32)
        sin = (X[1][0]).real.astype(np.float32)
        t_x = (X[2][0]).real.astype(np.float32)
        t_y = (X[3][0]).real.astype(np.float32)
        # scale = np.sqrt(np.square(cos) + np.square(sin))

        H = np.array([[cos, -sin, t_x], [sin, cos, t_y]])

        s = np.linalg.eigvals(H[:, :-1])
        R = s.max() / s.min()

        return H, R

    def crop_face(self, img, rect, margin=0.2):
        x1 = rect.left()
        x2 = rect.right()
        y1 = rect.top()
        y2 = rect.bottom()
        # size of face
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # add margin
        full_crop_x1 = x1 - int(w * margin)
        full_crop_y1 = y1 - int(h * margin)
        full_crop_x2 = x2 + int(w * margin)
        full_crop_y2 = y2 + int(h * margin)
        # size of face with margin
        new_size_w = full_crop_x2 - full_crop_x1 + 1
        new_size_h = full_crop_y2 - full_crop_y1 + 1

        # ensure that the region cropped from the original image with margin
        # doesn't go beyond the image size
        crop_x1 = max(full_crop_x1, 0)
        crop_y1 = max(full_crop_y1, 0)
        crop_x2 = min(full_crop_x2, img.shape[1] - 1)
        crop_y2 = min(full_crop_y2, img.shape[0] - 1)
        # size of the actual region being cropped from the original image
        crop_size_w = crop_x2 - crop_x1 + 1
        crop_size_h = crop_y2 - crop_y1 + 1

        # coordinates of region taken out of the original image in the new image
        new_location_x1 = crop_x1 - full_crop_x1
        new_location_y1 = crop_y1 - full_crop_y1
        new_location_x2 = crop_x1 - full_crop_x1 + crop_size_w - 1
        new_location_y2 = crop_y1 - full_crop_y1 + crop_size_h - 1

        new_img = np.random.randint(256, size=(new_size_h, new_size_w, img.shape[2])).astype('uint8')
        # new_img = np.random.rand(new_size_h, new_size_w, img.shape[2])

        new_img[new_location_y1: new_location_y2 + 1, new_location_x1: new_location_x2 + 1, :] = \
            img[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1, :]

        # if margin goes beyond the size of the image, repeat last row of pixels
        if new_location_y1 > 0:
            new_img[0:new_location_y1, :, :] = np.tile(new_img[new_location_y1, :, :], (new_location_y1, 1, 1))

        if new_location_y2 < new_size_h - 1:
            new_img[new_location_y2 + 1:new_size_h, :, :] = np.tile(new_img[new_location_y2:new_location_y2 + 1, :, :],
                                                                    (new_size_h - new_location_y2 - 1, 1, 1))
        if new_location_x1 > 0:
            new_img[:, 0:new_location_x1, :] = np.tile(new_img[:, new_location_x1:new_location_x1 + 1, :],
                                                       (1, new_location_x1, 1))
        if new_location_x2 < new_size_w - 1:
            # plt.imshow(new_img)
            new_img[:, new_location_x2 + 1:new_size_w, :] = np.tile(new_img[:, new_location_x2:new_location_x2 + 1, :],
                                                                    (1, new_size_w - new_location_x2 - 1, 1))

        return new_img


    def preprocess_input(self, img):

        # Expected input is BGR
        x = img - img.min()
        x = 255.0 * x / x.max()

        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68
        return x

    def run(self):
        Celebinfo = namedtuple('Celeb', ['filename', 'distance'])

        while not self.parent.isTerminated():

            while self.switching_model:
                self.recognition_running = False
                time.sleep(0.1)

            self.recognition_running = True

            faces = self.parent.getFaces()
            while faces == None:
                time.sleep(0.1)
                faces = self.parent.getFaces()

            validFaces = [f for f in faces if len(f['bboxes']) > self.minDetections]

            for face in validFaces:
                # get the timestamp of the most recent frame:
                timestamp = face['timestamps'][-1]
                unit = self.parent.getUnit(self, timestamp)

                if unit is not None:
                    img = unit.getFrame()
                    mean_box = np.mean(face['bboxes'], axis=0)
                    x, y, w, h = [int(c) for c in mean_box]

                    # Align the face to match the targets

                    # 1. DETECT LANDMARKS
                    dlib_box = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
                    dlib_img = img[..., ::-1].astype(np.uint8)
                    s = self.aligner(dlib_img, dlib_box)
                    landmarks = [[s.part(k).x, s.part(k).y] for k in range(s.num_parts)]

                    # 2. ALIGN
                    landmarks = np.array(landmarks)
                    M,R = self.estimateRigidTransform(landmarks, self.aligner_targets)

                    if R < 1.5:
                        crop = cv2.warpAffine(img, M, (224, 224), borderMode=2)
                    else:
                        # Seems to distort too much, probably error in landmarks, then let's just crop.
                        crop = self.crop_face(dlib_img, dlib_box)
                        crop = cv2.resize(crop, (224, 224))


                    siamese_target_size = self.siameseNet.input_shape[1:3]
                    crop_celeb = cv2.resize(crop, siamese_target_size)
                    if __debug__:
                        face["crop"] = crop_celeb
                    
                    crop = crop.astype(np.float32)
                    crop_celeb = crop_celeb.astype(np.float32) / 255.0

                    # # Recognize age
                    # Recognize only if new face or every 5 rounds
                    if "age" not in face or face["recog_round"] % 5 == 0:
                        agein = self.preprocess_input(crop) 
                        time_start = time.time()
                        ageout = self.ageNet.predict(np.expand_dims(agein, 0))[0]
                        age = np.dot(ageout, list(range(101)))

                        #with open("age_time.txt", "a") as fp:
                        #    fp.write("%.1f,%.8f\n" % (time.time(), elapsed_time))

                        if "age" in face:
                            face["age"] = 0.95 * face["age"] + 0.05 * age
                        else:
                            face["age"] = age
                            face["recog_round"] = 0

                        siamese_features = self.siameseNet.predict(crop_celeb[np.newaxis, ...])
                        K = 1  # This many nearest matches
                        celeb_distance, I = self.celeb_index.search(siamese_features, K)
                        celeb_idx = I[0][0]
                        celeb_filename = self.celeb_files[celeb_idx]

                        if "celebs" in face:
                            celebs = face["celebs"]
                            recognitions = celebs["recognitions"]

                            if recognitions < RecognitionThread.CELEB_RECOG_BUFFER:
                                celebs["indexes"].append(celeb_idx)
                            else:
                                celebs["indexes"][recognitions % RecognitionThread.CELEB_RECOG_BUFFER] = celeb_idx

                            celebs[celeb_idx] = Celebinfo(filename=celeb_filename, distance=celeb_distance)
                            celebs["recognitions"] += 1
                        else:
                            face["celebs"] = {
                                "indexes": [celeb_idx],
                                celeb_idx: Celebinfo(filename=celeb_filename, distance=celeb_distance),
                                "recognitions": 1}

                    # Switch to RGB here, because that is what these networks were trained with
                    nn_input = np.expand_dims(crop[..., ::-1]/255, axis=0)

                    # # Recognize gender                                       
                    # Recognize only if new face or every 6 rounds
                    # This makes it unlikely to have to recognize all 3 on the same frame
                    if "gender" not in face or face["recog_round"] % 6 == 0:
                        gender = self.genderNet.predict(nn_input)[0]
                        #print(gender)
                        #print("Gender time: {:.2f} ms".format(1000*(time.time() - time_start)))

#                        with open("gender_time.txt", "a") as fp:
#                            fp.write("%.1f,%.8f\n" % (time.time(), elapsed_time))
                        if "gender" in face:
                            face["gender"] = 0.8 * face["gender"] + 0.2 * gender
                        else:
                            face["gender"] = gender

                    # Recognize expression
                    # Recognize always as this is expected to change (ie. not constant)
                    out = self.expressionNet.predict(nn_input)

#                    with open("exp_time.txt", "a") as fp:
#                        fp.write("%.1f,%.8f\n" % (time.time(), elapsed_time))
                    #print("Expression time: {:.2f} ms".format(1000*(time.time() - time_start)))
                    t = out[0]
                    t = np.argmax(t)
                    expression = self.expressions[t]
                    face["expression"] = expression

                    face["recog_round"] += 1

    def switch_model(self, modelidx):

        self.siamesepath = self.siamesepaths[modelidx]

        print("Switching to", self.siamesepath)
        print("Stopping recognition thread...")
        self.switching_model = True

        # Wait for recognition thread to finish and stop before changing
        while self.recognition_running:
            time.sleep(0.1)

        self.initialize_celeb()

        print("Switching model complete. Resuming recognition thread...")
        self.switching_model = False

    def print_models(self):
        idx = 0
        while str(idx) in self.siamesepaths:
            desc = self.siamesepaths.get("{}_desc".format(idx), "")
            modelpath = self.siamesepaths[str(idx)]
            currentindicator = "<----- CURRENT MODEL" if modelpath == self.siamesepath else ""
            if desc:
                print("{}: {}, {} {}".format(idx, modelpath, desc, currentindicator))
            else:
                print("{}: {} {}".format(idx, modelpath, currentindicator))
            idx += 1


#!/usr/bin/env python

import threading
import time
import numpy as np
import os
import cv2
import dlib
import keras
from keras.utils.generic_utils import CustomObjectScope


class RecognitionThread(threading.Thread):
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
        agepath = params.get("recognition", "age_folder")
        with CustomObjectScope({'relu6': keras.layers.ReLU(6.),
                                'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
            self.ageNet = keras.models.load_model(os.path.join(agepath, 'model.h5'))
            self.ageNet._make_predict_function()

        ##### 2. GENDER
            genderpath = params.get("recognition", "gender_folder")
            self.genderNet = keras.models.load_model(os.path.join(genderpath, 'model.h5'))
            self.genderNet._make_predict_function()

        ##### 3. EXPRESSION
            expressionpath = params.get("recognition", "expression_folder")
            self.expressionNet = keras.models.load_model(os.path.join(expressionpath, 'model.h5'))
            self.expressionNet._make_predict_function()

        ##### Read class names

        self.expressions = []

        while True:
            try:
                key = "class%d" % len(self.expressions)
                name = params.get("expression", key)
                self.expressions.append(name)
            except:
                break

        self.minDetections = int(params.get("recognition", "mindetections"))

        # Starting the thread
        print("Recognition thread started...")

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

        while self.parent.isTerminated() == False:
            faces = self.parent.getFaces()
            while faces == None:
                time.sleep(0.1)
                faces = self.parent.getFaces()

            validFaces = [f for f in faces if len(f['bboxes']) > self.minDetections]
            recog_start = time.time()

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

                    crop = crop.astype(np.float32)

                    # # Recognize age
                    # Recognize only if new face or every 5 rounds

                    if "age" not in face or face["recog_round"] % 5 == 0:
                        agein = self.preprocess_input(crop) 
                        #time_start = time.time()
                        ageout = self.ageNet.predict(np.expand_dims(agein, 0))[0]
                        #print("Age time: {:.2f} ms".format(1000*(time.time() - time_start)))
                        age = np.dot(ageout, list(range(101)))

                        if "age" in face:
                            face["age"] = 0.75 * face["age"] + 0.25 * age
                        else:
                            face["age"] = age
                            face["recog_round"] = 0

                    # Switch to RGB here, because that is what these networks were trained with
                    nn_input = np.expand_dims(crop[..., ::-1]/255, axis=0)

                    # # Recognize gender                                       
                    # Recognize only if new face or every 6 rounds
                    # This makes it unlikely to have to recognize all 3 on the same frame
                    if "gender" not in face or face["recog_round"] % 6 == 0:
                        #time_start = time.time()
                        gender = self.genderNet.predict(nn_input)[0]
                        #print("Gender time: {:.2f} ms".format(1000*(time.time() - time_start)))
                        if "gender" in face:
                            face["gender"] = 0.8 * face["gender"] + 0.2 * gender
                        else:
                            face["gender"] = gender
                    

                    # Recognize expression
                    # Recognize always as this is expected to change (ie. not constant)
                    #time_start = time.time()
                    out = self.expressionNet.predict(nn_input)
                    #print("Expression time: {:.2f} ms".format(1000*(time.time() - time_start)))
                    t = out[0]
                    t = np.argmax(t)
                    expression = self.expressions[t]
                    face["expression"] = expression

                    face["recog_round"] += 1

            #print("Recognition loop time: {:.2f} ms".format(1000*(time.time() - recog_start)))
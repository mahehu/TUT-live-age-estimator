# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:44:40 2016

@author: agedemo
"""

from UnitServer import UnitServer
from GrabberThread import GrabberThread
from DetectionThread import DetectionThread
from RecognitionThread import RecognitionThread

import threading
import time
import datetime
import cv2
import copy
import os
import numpy as np
import subprocess

class ControllerThread(threading.Thread):
    """ Responsible for starting and shutting down all threads and
        services. """
        
    def __init__(self, params):
        threading.Thread.__init__(self)

        self.imageSaveTime = time.time()
        os.makedirs('saves', exist_ok=True)

        self.terminated = False
        self.caption = params.get("window", "caption")        

        self.minDetections = int(params.get("recognition", "mindetections"))
        
        self.displaysize = params.get("window", "displaysize")
        self.displaysize = self.displaysize.upper().split("X")
        self.displaysize = tuple([int(s) for s in self.displaysize])

        # Get current resolution
        self.resolution = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4', shell=True,
                                           stdout=subprocess.PIPE).communicate()[0].decode("utf-8").rstrip().split('x')
        self.resolution = [int(s) for s in self.resolution]

        # Start frame storage
        queueLength = params.getint("server", "num_frames")
        self.unitServer = UnitServer(queueLength)

        # Start Grabber thread
        self.grabberThread = GrabberThread(self, params)
        self.grabberThread.start()
        
        # Start Detection thread
        self.faces = []
        self.detectionThread = DetectionThread(self, params)
        self.detectionThread.start()

        # Start Recognition Thread
        self.recognitionThread = RecognitionThread(self, params)
        self.recognitionThread.start()

        unused_width = self.resolution[0] - self.displaysize[0]

        cv2.moveWindow(self.caption, unused_width//2, 0)  # Will move window when everything is running. Better way TODO
        self.commandInterface()

    def commandInterface(self):
        while True:
            text = input("Enter command (Q)uit, (L)ist models, (S)witch model: ").lower()
            if text == "l":
                self.recognitionThread.print_models()
            elif text == "s":
                idx = input("Please input a new index: ")
                try:
                    self.recognitionThread.switch_model(idx)
                except KeyError as e:
                    print("No such model index", e)

            elif text == "q":
                print("Bye!")
                self.terminate()
                break

    def run(self):
        while not self.terminated:
            time.sleep(0.5)
                        
    def putUnit(self, unit):
        
        # Show the newest frame immediately.
        self.showVideo(unit)
        
        # Send to further processing
        if not self.terminated:
            self.unitServer.putUnit(unit)
        
    def getUnit(self, caller, timestamp = None):
        
        return self.unitServer.getUnit(caller, timestamp)
    
    def terminate(self):
        
        self.terminated = True
       
    def drawBoundingBox(self, img, bbox):

        x,y,w,h = [int(c) for c in bbox]
        
        m = 0.2
        
        # Upper left corner
        pt1 = (x,y)
        pt2 = (int(x + m*w), y)
        cv2.line(img, pt1, pt2, color = [255,255,0], thickness = 2)

        pt1 = (x,y)
        pt2 = (x, int(y + m*h))
        cv2.line(img, pt1, pt2, color = [255,255,0], thickness = 2)

        # Upper right corner
        pt1 = (x + w, y)
        pt2 = (x + w, int(y + m*h))
        cv2.line(img, pt1, pt2, color = [255,255,0], thickness = 2)

        pt1 = (x + w, y)
        pt2 = (int(x + w - m * w), y)
        cv2.line(img, pt1, pt2, color = [255,255,0], thickness = 2)
        
        # Lower left corner
        pt1 = (x, y + h)
        pt2 = (x, int(y + h - m*h))
        cv2.line(img, pt1, pt2, color = [255,255,0], thickness = 2)

        pt1 = (x, y + h)
        pt2 = (int(x + m * w), y + h)
        cv2.line(img, pt1, pt2, color = [255,255,0], thickness = 2)
                
        # Lower right corner
        pt1 = (x + w, y + h)
        pt2 = (x + w, int(y + h - m*h))
        cv2.line(img, pt1, pt2, color = [255,255,0], thickness = 2)

        pt1 = (x + w, y + h)
        pt2 = (int(x + w - m * w), y + h)
        cv2.line(img, pt1, pt2, color = [255,255,0], thickness = 2)
        
    def drawFace(self, face, img):
        
        bbox = np.mean(face['bboxes'], axis = 0)
        
        self.drawBoundingBox(img, bbox)
        x, y, w, h = [int(c) for c in bbox]

        font_scale = 0.6

        # Check if text can overlap the celeb texts (goes past the bounding box), if so decrease size
        test_text = "FEMALE 100%"  # Probably longest text possible
        text_length = cv2.getTextSize(test_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
        if text_length > w:
            font_scale *= w/text_length

        # 1. AGE

        if "age" in face.keys():

            age = face['age']
            age_annotation = "Age: %.0f" % age
            txtLoc = (x, y + h + 30)

            cv2.putText(img,
                        text = age_annotation,
                        org = txtLoc,
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = font_scale,
                        color = [255,255,0],
                        thickness = 2)

        # 2. GENDER

        if "gender" in face.keys():
            
            gender = "MALE" if face['gender'] > 0.5 else "FEMALE"
            genderProb = max(face["gender"], 1-face["gender"])
            annotation = "%s %.0f %%" % (gender, 100.0 * genderProb)
            txtLoc = (x, y + h + 60)
    
            cv2.putText(img,
                        text = annotation,
                        org = txtLoc,
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = font_scale,
                        color = [255,255,0],
                        thickness = 2)

        # 3. EXPRESSION

        if "expression" in face.keys():

            expression = face["expression"]
            annotation = "%s" % (expression)
            txtLoc = (x, y + h + 90)
    
            cv2.putText(img,
                        text = annotation,
                        org = txtLoc,
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = font_scale,
                        color = [255,255,0],
                        thickness = 2)

        # 4. CELEBRITY TWIN

        # Clamp bounding box top to image
        y = 0 if y < 0 else y

        if "celebs" in face.keys():

            celebs = face["celebs"]
            indexes = celebs["indexes"]
            most_common = max(set(indexes), key=indexes.count)

            filename = celebs[most_common].filename
            distance = celebs[most_common].distance
            identity = filename.split(os.sep)[-2]

            celeb_img = cv2.imread(filename)
            aspect_ratio = celeb_img.shape[1] / celeb_img.shape[0]
            new_h = h
            new_w = int(aspect_ratio * h)
            try:
                celeb_img = cv2.resize(celeb_img, (new_w, new_h))
            except AssertionError:  # new_w or new_h is 0 ie bounding box size is 0
                return  # not a good way, this breaks if you add more functionality after celeb
                # TODO refactor into functions?

            # Cut out pixels overflowing image on the right
            x_end = x+w + new_w
            if x_end > img.shape[1]:
                remove_pixels = x_end - img.shape[1]
                celeb_img = celeb_img[:, :-remove_pixels, :]
                new_w -= remove_pixels

            # Cut out pixels overflowing image on the bottom
            y_end = y+new_h
            if y_end > img.shape[0]:
                remove_pixels = y_end - img.shape[0]
                celeb_img = celeb_img[:-remove_pixels, ...]
                new_h -= remove_pixels

            if celeb_img.size:
                img[y : y+new_h, x+w : x+w+new_w, :] = celeb_img
            
                annotation = "CELEBRITY"
                txtLoc = (x+w, y + h + 30)
                
                cv2.putText(img,
                            text = annotation,
                            org = txtLoc,
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = font_scale,
                            color = [255,255,0],
                            thickness = 2)
            
                annotation = "TWIN" # (%.0f %%)" % (100*np.exp(-face["celeb_distance"]))
                txtLoc = (x+w, y + h + 60)
                
                cv2.putText(img,
                            text = annotation,
                            org = txtLoc,
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = font_scale,
                            color = [255,255,0],
                            thickness = 2)
                
                annotation = identity.replace('ä', 'a').replace('ö', 'o').replace('å', 'o')
                txtLoc = (x+w, y + h + 90)
                
                cv2.putText(img,
                            text = annotation,
                            org = txtLoc,
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = font_scale,
                            color = [255,255,0],
                            thickness = 2)

        if __debug__:
            try:
                crop_128 = face["crop"]
                crop_h, crop_w = crop_128.shape[0:2]
                #if y+crop_h < img.shape[0] and x+w+new_w+crop_w < img.shape[1]:
                img[y : y+crop_h, x+w+new_w : x+w+new_w+crop_w, : ] = crop_128
            except:
                pass

            
    def showVideo(self, unit):
        
        unit.acquire()
        frame = copy.deepcopy(unit.getFrame())
        unit.release()
        
        # Annotate

        validFaces = [f for f in self.faces if len(f['bboxes']) > self.minDetections]

        for face in validFaces:
            self.drawFace(face, frame)
        
        frame = cv2.resize(frame, self.displaysize)
        cv2.imshow(self.caption, frame)
        key = cv2.waitKey(10)
        
        if key == 27:
            self.terminate()
            
    def findNearestFace(self, bbox):
        
        distances = []
        
        x,y,w,h = bbox
        bboxCenter = [x + w/2, y + h/2]
        
        for face in self.faces:
            
            x,y,w,h = np.mean(face['bboxes'], axis = 0)
            faceCenter = [x + w/2, y + h/2]
            
            distance = np.hypot(faceCenter[0] - bboxCenter[0], 
                                faceCenter[1] - bboxCenter[1])

            distances.append(distance)
        
        if len(distances) == 0:
            minIdx = None
            minDistance = None
        else:            
            minDistance = np.min(distances)
            minIdx = np.argmin(distances)

        return minIdx, minDistance        
        
    def setDetections(self, detections, timestamps):
        
        # Find the location among all recent face locations where this would belong

        for bbox, timestamp in zip(detections, timestamps):
            
            idx, dist = self.findNearestFace(bbox)

            if dist is not None and dist < 50:

                self.faces[idx]['bboxes'].append(bbox)
                self.faces[idx]['timestamps'].append(timestamp)
                
                if len(self.faces[idx]['bboxes']) > 7:
                    self.faces[idx]['bboxes'].pop(0)
                    self.faces[idx]['timestamps'].pop(0)
                    
            else:
                # This is a new face not in the scene before
                self.faces.append({'timestamps': [timestamp], 'bboxes': [bbox]})
        
        # Clean old detections:
        
        now = time.time()
        facesToRemove = []
        
        for i, face in enumerate(self.faces):
            if now - face['timestamps'][-1] > 0.5:
                facesToRemove.append(i)                
                    
        for i in facesToRemove:
            try:
                self.faces.pop(i)
            except:
                # Face was deleted by other thread. 
                pass
        
    def getFaces(self):
        
        if len(self.faces) == 0:
            return None
        else:
            return self.faces
    
    def isTerminated(self):
        
        return self.terminated
        

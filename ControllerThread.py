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
import sys
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

        self.terminated = False
        self.caption = params.get("window", "caption")

        self.initializeFonts(params)

        self.minDetections = int(params.get("recognition", "mindetections"))
        
        self.displaysize = params.get("window", "displaysize")
        self.displaysize = self.displaysize.upper().split("X")
        self.displaysize = tuple([int(s) for s in self.displaysize])

        self.debug = params.get("general", "debug") not in ("false", "False", "0")

        # Get current resolution (only implemented for Linux)
        if sys.platform == 'linux':
            self.resolution = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4', shell=True,
                                               stdout=subprocess.PIPE).communicate()[0].decode("utf-8").rstrip().split('x')
            self.resolution = [int(s) for s in self.resolution]
        else:
            self.resolution = None

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

        if self.resolution:
            unused_width = self.resolution[0] - self.displaysize[0]
            cv2.moveWindow(self.caption, unused_width//2, 0)  # Will move window to center after everything is running.

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

    def initializeFonts(self, params):
        """
        Tries to initialize freetype for nicer fonts, if not installed fall back to standard.
        Freetype isn't included in the PIP/Conda packages, so we can't require it.
        """
        self.freeType = None
        freetype_fontpath = params.get("window", "freetype_fontpath")
        sizetest_text = "FEMALE 100%"  # Probably longest text possible
        try:
            self.freeType = cv2.freetype.createFreeType2()
            self.freeType.loadFontData(fontFileName=freetype_fontpath, id=0)
            self.textBaseScale = 20  # Maximum text scale, will be decreased if there's overlap.
            self.textBaseWidth = self.freeType.getTextSize(sizetest_text, self.textBaseScale, -1)[0][0]

        except AttributeError:
            print("OpenCV Freetype not found, falling back to standard OpenCV font...")
            self.textBaseScale = 0.6
            self.textBaseWidth = cv2.getTextSize(sizetest_text, cv2.FONT_HERSHEY_SIMPLEX, self.textBaseScale, 2)[0][0]


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

    def AddCeleb(self, face, img, x, y, w, h):

        celebs = face["celebs"]
        indexes = celebs["indexes"]
        most_common = max(set(indexes), key=indexes.count)

        filename = celebs[most_common].filename
        distance = celebs[most_common].distance
        identity = filename.split(os.sep)[-2]

        celeb_img = cv2.imread(filename)
        aspect_ratio = celeb_img.shape[1] / celeb_img.shape[0]
        new_w = w
        new_h = int(w/aspect_ratio)
        if new_h > h:
            new_h = h
        try:
            celeb_img = cv2.resize(celeb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except AssertionError:  # new_w or new_h is 0 ie bounding box size is 0
            return None

        # Cut out pixels overflowing image on the right
        x_end = x + w + new_w
        if x_end > img.shape[1]:
            remove_pixels = x_end - img.shape[1]
            celeb_img = celeb_img[:, :-remove_pixels, :]
            new_w -= remove_pixels

        # Cut out pixels overflowing image on the bottom
        y_offset = h - new_h
        y_end = y + y_offset + new_h
        if y_end > img.shape[0]:
            remove_pixels = y_end - img.shape[0]
            celeb_img = celeb_img[:-remove_pixels, ...]
            new_h -= remove_pixels

        if celeb_img.size:
            img[y + y_offset: y + y_offset + new_h, x + w: x + w + new_w, :] = celeb_img
            return identity

    def drawFace(self, face, img):
        
        bbox = np.mean(face['bboxes'], axis = 0)
        
        self.drawBoundingBox(img, bbox)
        x, y, w, h = [int(c) for c in bbox]

        # 1. CELEBRITY TWIN

        celeb_identity = None

        # Clamp bounding box top to image
        y = 0 if y < 0 else y

        if "celebs" in face.keys():
            celeb_identity = self.AddCeleb(face, img, x, y, w, h)

        # Check if text can overlap the celeb texts (goes past the bounding box), if so decrease size
        text_size = self.textBaseScale

        if self.textBaseWidth > w:
            text_size *= w/self.textBaseWidth
            if self.freeType:
                text_size = int(text_size)  # Freetype doesn't accept float text size.


        # 1. AGE

        if "age" in face.keys():
            age = face['age']
            annotation = "Age: %.0f" % age
            txtLoc = (x, y + h + 30)
            self.writeText(img, annotation, txtLoc, text_size)


            # 2. GENDER

        if "gender" in face.keys():
            gender = "MALE" if face['gender'] > 0.5 else "FEMALE"
            genderProb = max(face["gender"], 1-face["gender"])
            annotation = "%s %.0f %%" % (gender, 100.0 * genderProb)
            txtLoc = (x, y + h + 60)
            self.writeText(img, annotation, txtLoc, text_size)

        # 3. EXPRESSION

        if "expression" in face.keys():
            expression = face["expression"]
            annotation = "%s" % (expression)
            txtLoc = (x, y + h + 90)
            self.writeText(img, annotation, txtLoc, text_size)

        if celeb_identity:
            annotation = "CELEBRITY"
            txtLoc = (x + w, y + h + 30)
            self.writeText(img, annotation, txtLoc, text_size)

            annotation = "TWIN"  # (%.0f %%)" % (100*np.exp(-face["celeb_distance"]))
            txtLoc = (x + w, y + h + 60)
            self.writeText(img, annotation, txtLoc, text_size)

            annotation = celeb_identity
            txtLoc = (x + w, y + h + 90)
            self.writeText(img, annotation, txtLoc, text_size)

        # DEBUG ONLY - Visualize aligned face crop in corner.
        if self.debug and "crop" in face.keys():
            croph, cropw = face["crop"].shape[0:2]
            imgh, imgw = img.shape[0:2]
            img[imgh-croph:, imgw-cropw:, :] = face["crop"][..., ::-1]

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

    def writeText(self, img, annotation, location, size):
        if self.freeType:
            self.freeType.putText(img=img,
                            text=annotation,
                            org=location,
                            fontHeight=size,
                            color=(255, 255, 0),
                            thickness=-1,
                            line_type=cv2.LINE_AA,
                            bottomLeftOrigin=True)
        else:
            annotation = annotation.replace('ä', 'a').replace('ö', 'o').replace('å', 'o')
            cv2.putText(img,
                        text=annotation,
                        org=location,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=size,
                        color=[255, 255, 0],
                        thickness=2)

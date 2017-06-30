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
import cv2
import copy
import numpy as np

class ControllerThread(threading.Thread):
    """ Responsible for starting and shutting down all threads and
        services. """
        
    def __init__(self, params):
        
        threading.Thread.__init__(self)

        self.terminated = False
        self.caption = params.get("window", "caption")        

        self.minDetections = int(params.get("recognition", "mindetections"))
        
        self.displaysize = params.get("window", "displaysize")
        self.displaysize = self.displaysize.upper().split("X")
        self.displaysize = tuple([int(s) for s in self.displaysize])
                
        # Start frame storage
        queueLength = params.getint("server", "num_frames")
        self.unitServer = UnitServer(queueLength)

        # Start Grabber thread
        self.grabberThread = GrabberThread(self, params)
        self.grabberThread.start()
        
        # Start Detection thread
        self.detectionThread = DetectionThread(self, params)
        self.detectionThread.start()
        self.faces = []
        
        # Start Recognition Thread
        self.recognitionThread = RecognitionThread(self, params)
        self.recognitionThread.start()
        
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
        x,y,w,h = [int(c) for c in bbox]

        # 1. AGE

        if "age" in face.keys():

            age = face['age']
            annotation = "Age: %.0f" % (age)
            txtLoc = (x, y + h + 30)
    
            cv2.putText(img,
                        text = annotation,
                        org = txtLoc,
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
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
                        fontScale = 1,
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
                        fontScale = 1,
                        color = [255,255,0],
                        thickness = 2)

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

            if dist is not None and dist < 100:

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
        
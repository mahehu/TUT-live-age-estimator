#!/usr/bin/env python

import cv2
import threading
import time
import numpy as np
import dlib
import GrabUnit

class DetectionThread(threading.Thread):

    def __init__(self, parent, params):

        threading.Thread.__init__(self)

        print "Initializing detection thread..."
        self.parent = parent

        # Create the haar cascade
        cascPath = params.get("detection", "cascade")
        self.detector = cv2.CascadeClassifier(cascPath)

        # Downsample factor to speed up detection
        self.scaling = float(params.get("detection", "scaling"))
        self.minSize = int(params.get("detection", "minsize"))

    def run(self):

        while self.parent.isTerminated() == False:

            unit = None

            while unit == None:

                unit = self.parent.getUnit(self)
                if unit == None:  # No units available yet
                    time.sleep(0.1)
                    
                if self.parent.isTerminated():
                    break

            if self.parent.isTerminated():
                break

            img = unit.getFrame()
            detection_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            unit.release()
            
            w = int(detection_img.shape[1] * self.scaling)
            h = int(detection_img.shape[0] * self.scaling)
            detection_img = cv2.resize(detection_img, (w, h))

            # Detect faces

            boxes = self.detector.detectMultiScale(
                detection_img,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(self.minSize, self.minSize),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
                        
            bboxes     = []
            timestamps = []
            
            for box in boxes:
                                
                box = [int(b / self.scaling) for b in box]
                bboxes.append(box)
                
                timestamps.append(unit.getTimeStamp())             

            self.parent.setDetections(bboxes, timestamps)
            

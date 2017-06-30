# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:44:40 2016

@author: agedemo
"""

import threading
import cv2
import GrabUnit

class GrabberThread(threading.Thread):

    def __init__(self, parent, params):
        
        threading.Thread.__init__(self)
        
        camId = params.getint("camera", "Id")
        
        camResolution = params.get("camera", "resolution")
        camResolution = camResolution.upper().split("X")
        camResolution = [int(x) for x in camResolution]
        print("Using camera %d at resolution %s" % (camId, camResolution))

        self.flipHor = params.getint("camera", "flip_horizontal")
        
        self.video = cv2.VideoCapture(camId)  # 0: Laptop camera, 1: USB-camera
        self.video.set(3, camResolution[0])  # 1280 #1920 Default: 640
        self.video.set(4, camResolution[1])  # 720  #1080 Default: 480

        self.parent = parent
        
        print("Grabber Thread initialized...")
        
    def run(self):

        while not self.parent.isTerminated():
            
            stat, frame = self.video.read()

            if frame is not None and not self.parent.isTerminated():
                if self.flipHor:
                    frame = frame[:, ::-1, ...]            

                unit = GrabUnit.GrabUnit(frame)
                
                self.parent.putUnit(unit)            
            
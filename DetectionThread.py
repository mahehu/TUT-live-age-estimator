#!/usr/bin/env python

import cv2
import threading
import time
import numpy as np
import GrabUnit

class DetectionThread(threading.Thread):

    def __init__(self, parent, params):

        threading.Thread.__init__(self)

        print("Initializing detection thread...")
        self.parent = parent

        frozen_graph = str(params.get("detection", "inference_graph"))
        text_graph = str(params.get("detection", "text_graph"))

        self.cvNet = cv2.dnn.readNetFromTensorflow(frozen_graph, text_graph)

	# Image input size, must match the network
        self.width = int(params.get("detection", "input_width"))
        self.height = int(params.get("detection", "input_height"))

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
        
            detection_img = img.copy()
            unit.release()

            rows, cols = img.shape[0:2]
            self.cvNet.setInput(cv2.dnn.blobFromImage(detection_img, size=(self.width, self.height),
                                                      swapRB=True, crop=False))
            timer = time.time()
            cvOut = self.cvNet.forward()

            #print("Det time: {:.2f} ms".format(1000*(time.time() - timer)))
            bboxes = []
            timestamps = []

            for detection in cvOut[0, 0, :, :]:
                score = float(detection[2])

                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)
                width = right - left
                height = bottom - top

                if score > 0.3 and width > 60:
                    bboxes.append([left, top, width, height])
                    timestamps.append(unit.getTimeStamp())

            self.parent.setDetections(bboxes, timestamps)

